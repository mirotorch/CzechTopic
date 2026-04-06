"""Trainer for Cross-Encoder model."""

import torch
import torch.nn as nn
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler


class CrossEncoderTrainer:
    def __init__(
        self,
        model,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 3e-5,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scaler = GradScaler("cuda")
        self.criterion = nn.BCELoss(reduction="none")

    def train_step(self, batch):
        self.model.train()
        self.optimizer.zero_grad()

        with autocast("cuda"):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                token_type_ids=batch["token_type_ids"],
            )

        logits = outputs["logits"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"]

        loss = self.criterion(logits, labels)
        loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        return {"loss": loss.item()}

    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            with autocast("cuda"):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    token_type_ids=batch["token_type_ids"],
                )

        logits = outputs["logits"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"]

        loss = self.criterion(logits, labels)
        loss = (loss * loss_mask).sum() / (loss_mask.sum() + 1e-8)

        return {
            "loss": loss.item(),
            "logits": logits,
            "labels": labels,
            "mask": loss_mask,
        }
