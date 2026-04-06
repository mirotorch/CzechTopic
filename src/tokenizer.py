"""Tokenizer for Cross-Encoder.

Tokenizes as: [CLS] topic_name [SEP] topic_description [SEP] text [SEP]
"""

import torch
from transformers import BertTokenizer


class CrossEncoderTokenizer:
    def __init__(self, tokenizer: BertTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
        
        self.sep_token = tokenizer.sep_token or "[SEP]"
        self.cls_token = tokenizer.cls_token or "[CLS]"
        self.pad_token_id = tokenizer.pad_token_id or 0

    def encode_single(
        self,
        topic_name: str,
        topic_description: str,
        text: str,
    ) -> dict:
        encoding = self.tokenizer(
            topic_name,
            topic_description,
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )
        
        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "offsets": encoding["offset_mapping"].squeeze(0),
            "topic_name": topic_name,
            "topic_description": topic_description,
        }

    def batch_encode(
        self,
        topics: list[dict],
        texts: list[str],
    ) -> dict:
        topic_names = [t["topic_name"] for t in topics]
        topic_descriptions = [t["topic_description"] for t in topics]
        
        encodings = self.tokenizer(
            topic_names,
            topic_descriptions,
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
            return_token_type_ids=True,
        )
        
        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "token_type_ids": encodings["token_type_ids"],
            "offsets": encodings["offset_mapping"],
        }


def offsets_to_word_labels(
    offsets: torch.Tensor,
    word_ids: list[int],
    labels: torch.Tensor,
) -> torch.Tensor:
    word_labels = torch.full((len(word_ids),), -100, dtype=torch.long)
    
    offset_np = offsets.cpu().numpy()
    label_np = labels.cpu().numpy()
    
    for token_idx, word_id in enumerate(word_ids):
        if word_id is None:
            continue
        if offset_np[token_idx][0] == 0 and offset_np[token_idx][1] == 0:
            continue
        word_labels[word_id] = label_np[token_idx]
    
    return word_labels
