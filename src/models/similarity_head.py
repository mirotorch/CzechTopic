"""Similarity Matrix Head for Cross-Encoder."""

import torch
import torch.nn as nn


class ClassificationHead(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(input_size, hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class SimilarityMatrixHead(nn.Module):
    def __init__(self, topic_size: int, text_size: int, aggregation: str = "max", dropout: float = 0.1):
        super().__init__()
        self.aggregation = aggregation
        self.topic_proj = nn.Linear(topic_size, text_size)
        self.classifier = ClassificationHead(text_size, text_size // 2, dropout)

    def forward(self, topic_hidden, text_hidden, topic_mask=None, text_mask=None):
        topic_proj = self.topic_proj(topic_hidden)
        sim = torch.matmul(topic_proj, text_hidden.transpose(-2, -1))

        if topic_mask is not None:
            topic_mask_expanded = topic_mask.unsqueeze(-1).float()
            sim = sim.masked_fill(topic_mask_expanded == 0, float('-inf'))
        
        if text_mask is not None:
            text_mask_expanded = text_mask.unsqueeze(1).float()
            sim = sim.masked_fill(text_mask_expanded == 0, float('-inf'))
        
        if self.aggregation == "max":
            sim_agg = sim.max(dim=1).values
        elif self.aggregation == "mean":
            if topic_mask is not None:
                topic_mask_sum = topic_mask.sum(dim=1, keepdim=True).clamp(min=1)
                sim_agg = sim.sum(dim=1) / topic_mask_sum
            else:
                sim_agg = sim.mean(dim=1)
        elif self.aggregation == "cls":
            sim_agg = sim[:, 0, :]
        else:
            sim_agg = sim.max(dim=1).values
        
        logits = self.classifier(sim_agg)
        return logits
