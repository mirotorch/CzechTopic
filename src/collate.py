"""Collate function for batching and subword-to-word alignment."""

import torch


def get_text_token_mask(input_ids, attention_mask, sep_token_id):
    """Create a mask for text tokens based on SEP positions.
    
    BERT format: [CLS] topic [SEP] text [SEP]
        - 2 SEPs total, text is between first and second SEP
    
    RoBERTa format: <s> topic </s></s> text </s>
        - 3 SEPs total, text starts after second SEP
    """
    batch_size, seq_len = input_ids.shape
    device = input_ids.device
    
    text_mask = torch.zeros_like(input_ids, dtype=torch.bool)
    
    for b in range(batch_size):
        sep_positions = (input_ids[b] == sep_token_id).nonzero(as_tuple=True)[0]
        num_seps = len(sep_positions)
        
        indices = torch.arange(seq_len, device=device)
        
        if num_seps == 0:
            pass
        elif num_seps == 1:
            first_sep = sep_positions[0].item()
            text_mask[b] = (indices > first_sep) & (attention_mask[b] == 1)
        elif num_seps == 2:
            first_sep = sep_positions[0].item()
            second_sep = sep_positions[1].item()
            text_mask[b] = (indices > first_sep) & (indices < second_sep) & (attention_mask[b] == 1)
        else:
            second_sep = sep_positions[1].item()
            last_sep = sep_positions[-1].item()
            text_mask[b] = (indices > second_sep) & (indices < last_sep) & (attention_mask[b] == 1)
    
    return text_mask


def collate_fn(batch, tokenizer, device):
    topics = [item["topic_name"] + " " + item["topic_description"] for item in batch]
    texts = [item["text"] for item in batch]

    encoded = tokenizer.batch_encode(topics, texts)

    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)
    offsets = encoded["offsets"]

    sep_token_id = tokenizer.tokenizer.sep_token_id
    is_text_token = get_text_token_mask(input_ids, attention_mask, sep_token_id)

    labels = []
    for i, item in enumerate(batch):
        item_labels = torch.zeros(input_ids.shape[1], dtype=torch.float)
        spans = [(ann["start"], ann["end"]) for ann in item["annotations"]]
        offset_np = offsets[i].numpy()

        for token_idx, (start, end) in enumerate(offset_np):
            if start == 0 and end == 0:
                continue

            if not is_text_token[i, token_idx]:
                continue

            for s_start, s_end in spans:
                if not (end <= s_start or start >= s_end):
                    item_labels[token_idx] = 1.0
                    break

        labels.append(item_labels)

    labels_tensor = torch.stack(labels).to(device)
    offsets_device = offsets.to(device)
    is_special = (
        (offsets_device == torch.zeros_like(offsets_device)).all(dim=-1).float()
    )
    loss_mask = is_text_token.float() * (1 - is_special)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels_tensor,
        "loss_mask": loss_mask,
        "offsets": offsets,
        "texts": texts,
    }
