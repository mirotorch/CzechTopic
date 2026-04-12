"""Configuration for Cross-Encoder model."""

from transformers import PretrainedConfig


class CrossEncoderConfig(PretrainedConfig):
    def __init__(self, num_labels: int = 1, dropout: float = 0.0, **kwargs):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.dropout = dropout
