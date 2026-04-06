"""Tokenizer for Cross-Encoder.

Tokenizes as: [CLS] topic_name topic_description [SEP] text [SEP]
"""

from transformers import BertTokenizer


class CrossEncoderTokenizer:
    def __init__(self, tokenizer: BertTokenizer, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def encode_single(
        self,
        topic_text: str,
        text: str,
    ) -> dict:
        encoding = self.tokenizer(
            topic_text,
            text,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        return {
            "input_ids": encoding["input_ids"].squeeze(0),
            "attention_mask": encoding["attention_mask"].squeeze(0),
            "token_type_ids": encoding["token_type_ids"].squeeze(0),
            "offsets": encoding["offset_mapping"].squeeze(0),
        }

    def batch_encode(
        self,
        topic_texts: list[str],
        texts: list[str],
    ) -> dict:
        encodings = self.tokenizer(
            topic_texts,
            texts,
            max_length=self.max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
            return_offsets_mapping=True,
        )

        return {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "token_type_ids": encodings["token_type_ids"],
            "offsets": encodings["offset_mapping"],
        }
