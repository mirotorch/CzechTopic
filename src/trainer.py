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
        accumulation_steps: int = 1,
    ):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.scaler = GradScaler("cuda")
        self.criterion = nn.BCELoss(reduction="none")
        self.accumulation_steps = accumulation_steps
        self.step_count = 0

    def train_step(self, batch):
        self.model.train()

        with autocast("cuda"):
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )

        logits = outputs["logits"]
        labels = batch["labels"]
        loss_mask = batch["loss_mask"]

        POSITIVE_WEIGHT = 15.0 
        weights = torch.ones_like(labels)
        weights[labels == 1.0] = POSITIVE_WEIGHT

        loss = self.criterion(logits, labels)
        
        loss = (loss * weights * loss_mask).sum() / (loss_mask.sum() + 1e-8)
        loss = loss / self.accumulation_steps

        self.scaler.scale(loss).backward()

        self.step_count += 1
        if self.step_count % self.accumulation_steps == 0:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()

        return {"loss": loss.item() * self.accumulation_steps}

    def eval_step(self, batch):
        self.model.eval()
        with torch.no_grad():
            with autocast("cuda"):
                outputs = self.model(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
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
