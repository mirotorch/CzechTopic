"""Prediction utilities for Cross-Encoder."""

import numpy as np
import torch

from .techniques import apply_nms


def predictions_to_spans(text, probs, offsets, threshold=0.5):
    if isinstance(offsets, torch.Tensor):
        offsets = offsets.numpy()
    if isinstance(probs, torch.Tensor):
        probs = probs.cpu().numpy()

    if len(probs) != len(offsets):
        min_len = min(len(probs), len(offsets))
        probs = probs[:min_len]
        offsets = offsets[:min_len]

    words = []
    word_probs = []
    current_word_start = None
    current_word_end = None
    current_word_probs = []
    prev_end = None

    for token_idx in range(len(offsets)):
        start, end = offsets[token_idx]
        
        if start == 0 and end == 0:
            continue

        is_cont = (start == prev_end) if prev_end is not None else False

        if current_word_start is not None and is_cont:
            current_word_probs.append(probs[token_idx])
            current_word_end = end
        else:
            if current_word_start is not None:
                words.append((current_word_start, current_word_end))
                word_probs.append(max(current_word_probs) if current_word_probs else 0.0)

            current_word_start = start
            current_word_end = end
            current_word_probs = [probs[token_idx]]
        
        prev_end = end

    if current_word_start is not None:
        words.append((current_word_start, current_word_end))
        word_probs.append(max(current_word_probs) if current_word_probs else 0.0)

    word_mask = [1 if p >= threshold else 0 for p in word_probs]

    spans = []
    i = 0
    while i < len(word_mask):
        if word_mask[i] == 1:
            start = words[i][0]
            end = words[i][1]
            while i + 1 < len(word_mask) and word_mask[i + 1] == 1:
                i += 1
                end = words[i][1]
            if end > start:
                spans.append((start, end))
        i += 1

    annotations = []
    for start, end in spans:
        text_piece = text[start:end]
        annotations.append({
            "start": int(start),
            "end": int(end),
            "text_piece": text_piece
        })

    return annotations


def span_predictions_to_spans(text, span_scores, span_indices, offsets, threshold=0.5, max_char_gap=5):
    if hasattr(offsets, "cpu"):
        offsets = offsets.cpu().numpy()
    if hasattr(span_scores, "cpu"):
        span_scores = span_scores.detach().cpu().numpy()
    if hasattr(span_indices, "cpu"):
        span_indices = span_indices.detach().cpu().numpy()

    candidate_spans = []
    candidate_scores = []

    for (start_idx, end_idx), score in zip(span_indices, span_scores):
        if score < threshold:
            continue
            
        start_char, _ = offsets[start_idx]
        _, end_char = offsets[end_idx]

        if (start_char == 0 and end_char == 0) or end_char <= start_char:
            continue

        while start_char > 0 and text[start_char - 1].isalnum():
            start_char -= 1
        while end_char < len(text) and text[end_char].isalnum():
            end_char += 1

        candidate_spans.append((int(start_char), int(end_char)))
        candidate_scores.append(float(score))

    selected_spans = apply_nms(candidate_spans, candidate_scores, threshold=threshold)
    selected_spans.sort(key=lambda x: x[0][0])

    annotations = []
    
    # Gap Bridging (to fix fragmentation)
    if selected_spans:
        current_start, current_end = selected_spans[0][0]
        
        for i in range(1, len(selected_spans)):
            next_start, next_end = selected_spans[i][0]
            
            if next_start - current_end <= max_char_gap:
                current_end = max(current_end, next_end)
            else:
                extracted_string = text[current_start:current_end].strip()
                if extracted_string:
                    real_start = text.find(extracted_string, current_start)
                    real_end = real_start + len(extracted_string)
                    
                    annotations.append({
                        "start": real_start,
                        "end": real_end,
                        "text_piece": extracted_string,
                    })
                current_start, current_end = next_start, next_end
                
        extracted_string = text[current_start:current_end].strip()
        if extracted_string:
            real_start = text.find(extracted_string, current_start)
            real_end = real_start + len(extracted_string)
            annotations.append({
                "start": real_start,
                "end": real_end,
                "text_piece": extracted_string,
            })

    return annotations


def calculate_word_f1(pred_masks, gt_masks):
    tp = fp = fn = 0
    for pred, gt in zip(pred_masks, gt_masks):
        tp += int(((pred == 1) & (gt == 1)).sum())
        fp += int(((pred == 1) & (gt == 0)).sum())
        fn += int(((pred == 0) & (gt == 1)).sum())

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}
