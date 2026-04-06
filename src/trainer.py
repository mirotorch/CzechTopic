"""Trainer for Cross-Encoder model."""

import torch
import torch.nn as nn

from .models import TopicCrossEncoder
from .tokenizer import CrossEncoderTokenizer


class CrossEncoderTrainer:
    def __init__(
        self,
        model: TopicCrossEncoder,
        tokenizer: CrossEncoderTokenizer,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        lr: float = 2e-5,
    ):
        self.model = model.to(device)
        self.tokenizer = tokenizer
        self.device = device
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
        self.criterion = nn.BCEWithLogitsLoss(reduction="none")

    def train_step(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        word_labels,
        offsets,
    ):
        self.model.train()
        self.optimizer.zero_grad()
        
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
        )
        
        logits = outputs["logits"]
        
        batch_size, seq_len = logits.shape
        labels = torch.zeros(batch_size, seq_len, device=self.device)
        for i, (off, wl) in enumerate(zip(offsets, word_labels)):
            word_id = 0
            for token_idx, (start, end) in enumerate(off.tolist()):
                if start == 0 and end == 0:
                    continue
                if word_id < len(wl):
                    labels[i, token_idx] = wl[word_id]
                word_id += 1
        
        mask = (labels != -100).float()
        active_logits = logits[mask.bool()]
        active_labels = labels[mask.bool()]
        
        loss = self.criterion(active_logits, active_labels).mean()
        loss.backward()
        self.optimizer.step()
        
        return {"loss": loss.item()}

    def predict(
        self,
        input_ids,
        attention_mask,
        token_type_ids,
        offsets,
    ) -> list[list[int]]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
            )
        
        logits = outputs["logits"]
        probs = torch.sigmoid(logits)
        predictions = (probs > 0.5).long().cpu().numpy()
        
        results = []
        for pred, off in zip(predictions, offsets):
            word_preds = []
            word_id = 0
            for token_idx, (start, end) in enumerate(off.tolist()):
                if start == 0 and end == 0:
                    continue
                word_preds.append(int(pred[token_idx]))
                word_id += 1
            results.append(word_preds)
        
        return results
