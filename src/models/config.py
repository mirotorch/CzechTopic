"""Configuration for the Cross-Encoder model."""

from transformers import BertConfig


class CrossEncoderConfig(BertConfig):
    def __init__(
        self,
        num_labels: int = 2,
        topic_hidden_size: int = 256,
        dropout: float = 0.1,
        similarity_aggregation: str = "max",
        **kwargs
    ):
        super().__init__(**kwargs)
        self.num_labels = num_labels
        self.topic_hidden_size = topic_hidden_size
        self.dropout = dropout
        self.similarity_aggregation = similarity_aggregation
