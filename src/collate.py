"""Collate function for batching and subword-to-word alignment."""

import torch


def collate_fn(batch, tokenizer, device):
    topics = [item["topic_name"] + " " + item["topic_description"] for item in batch]
    texts = [item["text"] for item in batch]

    encoded = tokenizer.batch_encode(topics, texts)

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    token_type_ids = encoded["token_type_ids"].to(device)
    offsets = encoded["offsets"]

    labels = []
    for i, item in enumerate(batch):
        item_labels = torch.zeros(input_ids.shape[1], dtype=torch.float)
        spans = [(ann["start"], ann["end"]) for ann in item["annotations"]]
        offset_np = offsets[i].numpy()

        for token_idx, (start, end) in enumerate(offset_np):
            if start == 0 and end == 0:
                continue

            is_text_token = token_type_ids[i, token_idx] == 1
            if not is_text_token:
                continue

            for s_start, s_end in spans:
                if not (end <= s_start or start >= s_end):
                    item_labels[token_idx] = 1.0
                    break

        labels.append(item_labels)

    labels_tensor = torch.stack(labels).to(device)
    is_text_token = (token_type_ids == 1).float()
    offsets_device = offsets.to(device)
    is_special = (
        (offsets_device == torch.zeros_like(offsets_device)).all(dim=-1).float()
    )
    loss_mask = is_text_token * (1 - is_special)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "labels": labels_tensor,
        "loss_mask": loss_mask,
        "offsets": offsets,
        "texts": texts,
    }
