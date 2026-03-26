"""Common utilities for evaluation scripts.

Author: Martin Kostelník (ikostelnik@fit.vut.cz)
Date: 11.3.2026
"""

import json
from pathlib import Path
import re
from typing import Sequence

import pandas as pd
import numpy as np


METRIC_COLS = [
    "text_precision", "text_recall", "text_f1", "text_accuracy",
    "span_precision", "span_recall", "span_f1", "span_iou",
]


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r") as f:
        return [json.loads(line) for line in f]
    

def split_text_into_words(text: str) -> list[tuple[str, int, int]]:
    """
    Returns list of (word, start_char, end_char)
    """
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"\w+(?:'\w+)?|[^\w\s]", text)]


def char_spans_to_word_mask(
    text: str,
    spans: list[tuple[int, int]],
) -> tuple[list[str], list[int]]:
    """
    spans: list of (start, end), end is exclusive

    Returns:
      words: list of words
      mask:  list of 0/1, same length as words
    """
    words_with_offsets = split_text_into_words(text)

    mask = []
    for _, w_start, w_end in words_with_offsets:
        in_span = 0
        for s_start, s_end in spans:
            # overlap check
            if not (w_end <= s_start or w_start >= s_end):
                in_span = 1
                break
        mask.append(in_span)

    words = [w for w, _, _ in words_with_offsets]
    return words, mask


def create_labels(items: list[dict]) -> list[np.ndarray]:
    all_labels = []
    for item in items:
        text = item["text"]
        spans = [(ann["start"], ann["end"]) for ann in item["annotations"]]
        _, mask = char_spans_to_word_mask(text, spans)
    
        all_labels.append(np.array(mask, dtype=int))

    return all_labels


def calculate_pair_metrics(
    predictions: list[dict],
    ground_truth: list[dict],
    rater_1: str | None = None,
    rater_2: str | None = None,
    against_p1: bool = False
) -> pd.DataFrame:
    all_pred_labels = create_labels(predictions)
    all_gt_labels = create_labels(ground_truth)

    detailed_results = []
    for gt_labels, pred_labels, gt_item, pred_item in zip(all_gt_labels, all_pred_labels, ground_truth, predictions, strict=True):
        assert pred_item["text_id"] == gt_item["text_id"] and pred_item["topic_id"] == gt_item["topic_id"] and pred_item["cluster_id"] == gt_item["cluster_id"], "Mismatched text_id or topic_id or cluster_id between predictions and ground truth"

        is_negative_sample = sum(gt_labels) == 0 and sum(pred_labels) == 0

        TP = sum((gt_labels == 1) & (pred_labels == 1))
        FP = sum((gt_labels == 0) & (pred_labels == 1))
        FN = sum((gt_labels == 1) & (pred_labels == 0))
        TN = sum((gt_labels == 0) & (pred_labels == 0))

        # Calculate binary classification
        binary_pred = sum(pred_labels) > 0
        binary_gt = sum(gt_labels) > 0
        binary = int(binary_pred == binary_gt)

        precision = TP / (TP + FP) if (TP + FP) > 0 else 1.0
        recall = TP / (TP + FN) if (TP + FN) > 0 else 1.0
        f1 = 2 * TP / (2 * TP + FP + FN) if (2 * TP + FP + FN) > 0 else 1.0
        accuracy = (TP + TN) / (TP + FP + FN + TN)
        iou = TP / (TP + FP + FN) if (TP + FP + FN) > 0 else 1.0

        detailed_result = {
            "text_id": gt_item["text_id"],
            "topic_id": gt_item["topic_id"],
            "rater_1_id": rater_1,
            "rater_2_id": rater_2,
            "against_p1": against_p1,
            "cluster_id": gt_item["cluster_id"],
            "is_negative_sample": is_negative_sample,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "accuracy": accuracy,
            "iou": iou,
            "binary": binary,
            "succ@iou_0.5": int(iou >= 0.5),
            "succ@iou_0.75": int(iou >= 0.75),
            "succ@iou_0.9": int(iou >= 0.9),
            "tp": int(TP),
            "fp": int(FP),
            "fn": int(FN),
            "tn": int(TN),
            "len": len(gt_labels),
        }
        detailed_results.append(detailed_result)

    return pd.DataFrame(detailed_results)


def align_data(
    gt: list[dict],
    pred: list[dict],
) -> tuple[list[dict], list[dict]]:
    gt_keys = [(item["text_id"], item["topic_id"]) for item in gt]
    pred_keys = [(item["text_id"], item["topic_id"]) for item in pred]
    common_keys = set(gt_keys) & set(pred_keys)
    gt_ = [item for item in gt if (item["text_id"], item["topic_id"]) in common_keys]
    pred_ = [item for item in pred if (item["text_id"], item["topic_id"]) in common_keys]
    gt_.sort(key=lambda x: (x["text_id"], x["topic_id"]))
    pred_.sort(key=lambda x: (x["text_id"], x["topic_id"]))
    return gt_, pred_


def safe_div(num: np.ndarray, den: np.ndarray) -> np.ndarray:
    num = np.asarray(num, dtype=float)
    den = np.asarray(den, dtype=float)
    out = np.full_like(num, np.nan, dtype=float)
    np.divide(num, den, out=out, where=(den != 0))
    return out


def metrics_from_counts(tp, fp, fn, tn=None) -> dict[str, np.ndarray]:
    tp = np.asarray(tp, dtype=float)
    fp = np.asarray(fp, dtype=float)
    fn = np.asarray(fn, dtype=float)

    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * tp, 2 * tp + fp + fn)
    iou = safe_div(tp, tp + fp + fn)

    out = {"precision": precision, "recall": recall, "f1": f1, "iou": iou}
    if tn is not None:
        tn = np.asarray(tn, dtype=float)
        acc = safe_div(tp + tn, tp + fp + fn + tn)
        out["accuracy"] = acc
    return out


def aggregate_counts(df: pd.DataFrame, group_cols: Sequence[str]) -> pd.DataFrame:
    return (
        df.groupby(list(group_cols), dropna=False)[
            ["tp", "fp", "fn", "tn", "txt_tp", "txt_fp", "txt_fn", "txt_tn"]
        ]
        .sum()
        .reset_index()
    )


def attach_metrics(agg: pd.DataFrame) -> pd.DataFrame:
    agg = agg.copy()

    sm = metrics_from_counts(agg["tp"], agg["fp"], agg["fn"], tn=None)
    agg["span_precision"] = sm["precision"]
    agg["span_recall"] = sm["recall"]
    agg["span_f1"] = sm["f1"]
    agg["span_iou"] = sm["iou"]

    tm = metrics_from_counts(agg["txt_tp"], agg["txt_fp"], agg["txt_fn"], tn=agg["txt_tn"])
    agg["text_precision"] = tm["precision"]
    agg["text_recall"] = tm["recall"]
    agg["text_f1"] = tm["f1"]
    agg["text_accuracy"] = tm["accuracy"]

    return agg


def topic_scores_fixed_rater1_vs_group(df: pd.DataFrame, compare_col: str, metric_cols: list[str] = METRIC_COLS) -> pd.DataFrame:
    agg = aggregate_counts(df, ["topic_id", compare_col])
    agg = attach_metrics(agg)
    grouped = agg.groupby("topic_id", dropna=False)[metric_cols].mean().reset_index()
    return grouped


def bootstrap_topic_macro_ci_df(
    topic_scores: pd.DataFrame,
    n_boot: int,
    seed: int,
    metric_cols: Sequence[str] = METRIC_COLS,
) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    ts = topic_scores.groupby("topic_id", as_index=False)[list(metric_cols)].mean()
    point = ts[list(metric_cols)].mean(axis=0)

    M = ts[list(metric_cols)].to_numpy(dtype=float)
    n_topics = M.shape[0]

    boot = np.empty((n_boot, len(metric_cols)), dtype=float)
    for b in range(n_boot):
        idx = rng.integers(0, n_topics, size=n_topics)
        boot[b, :] = np.nanmean(M[idx, :], axis=0)

    lo = np.nanpercentile(boot, 2.5, axis=0)
    hi = np.nanpercentile(boot, 97.5, axis=0)

    return pd.DataFrame({
        "metric": list(metric_cols),
        "point": point.to_numpy(dtype=float),
        "ci_low_95": lo,
        "ci_high_95": hi,
    })


def derive_text_presence_confusion(df: pd.DataFrame) -> pd.DataFrame:
    """
    From word-level counts derive text-level topic presence confusion:
      gold_present (rater_1): (tp+fn) > 0
      pred_present (rater_2): (tp+fp) > 0
    """
    df = df.copy()
    gold_present = (df["tp"] + df["fn"]) > 0
    pred_present = (df["tp"] + df["fp"]) > 0

    df["txt_tp"] = (gold_present & pred_present).astype(int)
    df["txt_fp"] = ((~gold_present) & pred_present).astype(int)
    df["txt_fn"] = (gold_present & (~pred_present)).astype(int)
    df["txt_tn"] = ((~gold_present) & (~pred_present)).astype(int)
    return df