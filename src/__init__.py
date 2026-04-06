"""Cross-Encoder for Topic Localization in Czech Texts.

Tokenizes as: [CLS] topic_name [SEP] topic_description [SEP] text [SEP]
Runs BERT once, then computes similarity matrix between topic and text tokens.
"""

import torch
from transformers import BertModel, BertTokenizer

from .models import CrossEncoderConfig, TopicCrossEncoder
from .tokenizer import CrossEncoderTokenizer
from .trainer import CrossEncoderTrainer

__all__ = [
    "CrossEncoderConfig",
    "TopicCrossEncoder",
    "CrossEncoderTokenizer",
    "CrossEncoderTrainer",
    "BertModel",
    "BertTokenizer",
]


if __name__ == "__main__":
    tokenizer = BertTokenizer.from_pretrained("bert-base-multilingual-cased")
    tokenizer_wrapper = CrossEncoderTokenizer(tokenizer, max_length=512)
    
    config = CrossEncoderConfig(
        vocab_size=tokenizer.vocab_size,
        hidden_size=768,
        num_attention_heads=12,
        num_hidden_layers=6,
        intermediate_size=3072,
    )
    model = TopicCrossEncoder(config)
    
    topic = {
        "topic_name": "Pracovni spory",
        "topic_description": "Stávky a konflikty mezi dělníky",
    }
    text = "Testovací text o stávce haviřů v hornictví."
    
    encoded = tokenizer_wrapper.encode_single(
        topic_name=topic["topic_name"],
        topic_description=topic["topic_description"],
        text=text,
    )
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    outputs = model(
        input_ids=encoded["input_ids"].unsqueeze(0).to(device),
        attention_mask=encoded["attention_mask"].unsqueeze(0).to(device),
        token_type_ids=encoded["token_type_ids"].unsqueeze(0).to(device),
    )
    
    print(f"Output logits shape: {outputs['logits'].shape}")
