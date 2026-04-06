"""CzechTopic Cross-Encoder implementation.

Architecture:
- [CLS] topic_name topic_description [SEP] text [SEP]
- Single BERT forward pass
- Dot product similarity between topic and text tokens
- Max pooling over topic tokens
- Sigmoid activation
"""

from .config import CrossEncoderConfig
from .model import TopicCrossEncoder
from .tokenizer import CrossEncoderTokenizer

__all__ = ["CrossEncoderConfig", "TopicCrossEncoder", "CrossEncoderTokenizer"]
