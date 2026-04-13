"""Main training script for CzechTopic Cross-Encoder."""

import argparse
import json
import logging
import random
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoConfig, AutoModel, AutoTokenizer

import wandb

from .collate import collate_fn
from .config import CrossEncoderConfig
from .model import TopicCrossEncoder
from .predict import calculate_word_f1, predictions_to_spans
from .techniques import TECHNIQUE_FACTORIES
from .tokenizer import CrossEncoderTokenizer
from .trainer import CrossEncoderTrainer

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_data(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_checkpoint_path(output_dir: Path, technique: str, model_name: str) -> Path:
    safe_model_name = model_name.replace("/", "-")
    return output_dir / f"best_model_{technique}_{safe_model_name}.pt"


class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("dataset/dev-dataset"))
    parser.add_argument("--output-dir", type=Path, default=Path("output"))
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--gradient-accumulation", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--patience", type=int, default=1)
    parser.add_argument(
        "--technique",
        type=str,
        default="max",
        choices=sorted(TECHNIQUE_FACTORIES),
        help="Pooling technique defined in techniques.py",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="bert-base-multilingual-cased",
        help="Pretrained model name (e.g., bert-base-multilingual-cased, robeczech)",
    )
    args = parser.parse_args()

    wandb.init(
        project="czech-topic-cross-encoder",  # Name of your project in W&B
        config=vars(args),  # Automatically logs all your hyperparameters!
        name=f"lr-{args.lr}_bs-{args.batch_size}",  # Gives the run a readable name
    )
    set_seed(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_data = load_data(args.data_dir / "train.jsonl")
    val_data = load_data(args.data_dir / "val.jsonl")
    logger.info(f"Loaded {len(train_data)} train, {len(val_data)} val samples")

    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer_wrapper = CrossEncoderTokenizer(tokenizer, max_length=args.max_length)

    pretrained_bert = AutoModel.from_pretrained(args.model_name)

    config_dict = pretrained_bert.config.to_dict()
    config_dict["sep_token_id"] = tokenizer.sep_token_id
    config = CrossEncoderConfig(**config_dict)
    model = TopicCrossEncoder(config, technique=args.technique, encoder=pretrained_bert)
    checkpoint_path = get_checkpoint_path(args.output_dir, args.technique, args.model_name)

    logger.info(
        f"Model initialized with {sum(p.numel() for p in model.parameters()):,} parameters"
    )
    logger.info(f"Using pooling technique: {args.technique}")

    trainer = CrossEncoderTrainer(
        model, device=device, lr=args.lr, accumulation_steps=args.gradient_accumulation
    )

    train_dataset = Dataset(train_data)
    val_dataset = Dataset(val_data)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=lambda b: collate_fn(b, tokenizer_wrapper, device),
    )

    best_f1 = 0.0
    patience_counter = 0
    best_threshold = 0.5

    for epoch in range(args.epochs):
        logger.info(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_losses = []
        for batch in tqdm(train_loader, desc="Training"):
            loss_dict = trainer.train_step(batch)
            train_losses.append(loss_dict["loss"])

        avg_train_loss = np.mean(train_losses)
        logger.info(f"Train loss: {avg_train_loss:.4f}")

        wandb.log({"train/loss": avg_train_loss, "epoch": epoch + 1}, commit=False)

        val_logits = []
        val_labels = []
        val_masks = []
        val_texts = []
        val_offsets = []

        val_loader = DataLoader(
            val_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            collate_fn=lambda b: collate_fn(b, tokenizer_wrapper, device),
        )

        for batch in tqdm(val_loader, desc="Validation"):
            out = trainer.eval_step(batch)
            val_logits.append(out["logits"].cpu())
            val_labels.append(out["labels"].cpu())
            val_masks.append(out["mask"].cpu())
            val_texts.extend(batch["texts"])
            val_offsets.extend(batch["offsets"])

        val_logits = torch.cat(val_logits, dim=0)
        val_labels = torch.cat(val_labels, dim=0)
        val_masks = torch.cat(val_masks, dim=0)

        thresholds = [i * 0.05 for i in range(1, 20)]
        best_val_f1 = 0.0
        best_epoch_threshold = 0.5

        for thresh in thresholds:
            pred_word_masks = []
            gt_word_masks = []

            for i in range(val_logits.shape[0]):
                mask = val_masks[i].numpy()
                probs = val_logits[i].numpy()
                labels = val_labels[i].numpy()

                valid_mask = mask > 0.5
                valid_probs = probs[valid_mask]
                valid_labels = labels[valid_mask]

                pred_word_masks.append((valid_probs >= thresh).astype(int))
                gt_word_masks.append(valid_labels.astype(int))

            metrics = calculate_word_f1(pred_word_masks, gt_word_masks)
            f1 = metrics["f1"]

            if f1 > best_val_f1:
                best_val_f1 = f1
                best_epoch_threshold = thresh

        logger.info(
            f"Val word-F1: {best_val_f1:.4f} @ threshold={best_epoch_threshold:.2f}"
        )

        wandb.log(
            {
                "val/f1": best_val_f1,
                "val/best_threshold": best_epoch_threshold,
                "epoch": epoch + 1,
            }
        )

        if best_val_f1 > best_f1:
            best_f1 = best_val_f1
            best_threshold = best_epoch_threshold
            patience_counter = 0
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(
                f"Saved best model to {checkpoint_path.name} (F1={best_f1:.4f}), "
                f"best threshold is {best_threshold}"
            )
        else:
            patience_counter += 1
            logger.info(f"No improvement. Patience: {patience_counter}/{args.patience}")

        if patience_counter >= args.patience:
            logger.info("Early stopping triggered")
            break

    logger.info(f"\nTraining complete. Best val F1: {best_f1:.4f}")

    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    wandb.finish()


if __name__ == "__main__":
    main()
