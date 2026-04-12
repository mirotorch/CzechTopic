"""Run evaluation on test set using trained model."""

import argparse
import json
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import BertTokenizer

from .config import CrossEncoderConfig
from .model import TopicCrossEncoder
from .predict import predictions_to_spans
from .techniques import TECHNIQUE_FACTORIES
from .tokenizer import CrossEncoderTokenizer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--output-path", type=Path, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument(
        "--technique",
        type=str,
        default="max",
        choices=sorted(TECHNIQUE_FACTORIES),
        help="Pooling technique defined in techniques.py",
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer_wrapper = CrossEncoderTokenizer(tokenizer, max_length=args.max_length)

    config = CrossEncoderConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=4,
        intermediate_size=3072,
    )
    model = TopicCrossEncoder(config, technique=args.technique)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print("Model loaded")

    test_data = []
    for jsonl_path in sorted(args.data_dir.glob("*.jsonl")):
        if jsonl_path.name == "annotators.csv":
            continue
        with open(jsonl_path) as f:
            test_data.extend([json.loads(line) for line in f])
    print(f"Loaded {len(test_data)} test samples")

    predictions = []
    for item in tqdm(test_data):
        encoded = tokenizer_wrapper.encode_single(
            item["topic_name"] + " " + item["topic_description"], item["text"]
        )

        with torch.no_grad():
            inputs = {
                "input_ids": encoded["input_ids"].unsqueeze(0).to(device),
                "attention_mask": encoded["attention_mask"].unsqueeze(0).to(device),
                "token_type_ids": encoded["token_type_ids"].unsqueeze(0).to(device),
            }
            out = model(**inputs)
            probs = out["logits"].squeeze(0).cpu().numpy()

        annotations = predictions_to_spans(
            item["text"], probs, encoded["offsets"].numpy(), threshold=args.threshold
        )

        predictions.append(
            {
                "text_id": item["text_id"],
                "topic_id": item["topic_id"],
                "cluster_id": item["cluster_id"],
                "text": item["text"],
                "annotations": annotations,
            }
        )

    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        for pred in predictions:
            f.write(json.dumps(pred) + "\n")

    print(f"Predictions saved to {args.output_path}")


if __name__ == "__main__":
    main()
