"""Compare predictions against annotator JSONL files and print mismatches."""

import argparse
import json
import random
import sys
from pathlib import Path


def load_jsonl(path: Path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def get_item_key(item):
    return (item["text_id"], item["topic_id"])


def normalize_annotations(item):
    annotations = item.get("annotations", [])
    normalized = []
    for ann in annotations:
        normalized.append(
            {
                "start": ann["start"],
                "end": ann["end"],
                "text_piece": ann["text_piece"],
            }
        )
    return normalized


def format_error_analysis(text, true_annotations, predicted_annotations, metadata=None):
    """
    Compares true annotations vs predictions and returns a readable mismatch report.
    """
    true_texts = [ann["text_piece"] for ann in true_annotations]
    pred_texts = [ann["text_piece"] for ann in predicted_annotations]

    if set(true_texts) == set(pred_texts):
        return None

    lines = ["", "=" * 80, "ERROR DETECTED", "-" * 80]
    if metadata is not None:
        lines.append(
            "ITEM: "
            f"text_id={metadata['text_id']}, topic_id={metadata['topic_id']}, "
            f"cluster_id={metadata['cluster_id']}"
        )
        lines.append(f"TOPIC: {metadata['topic_name']}")
    lines.append(f"ORIGINAL TEXT:\n{text}\n")

    missed = [t for t in true_texts if t not in pred_texts]
    if missed:
        lines.append("FALSE NEGATIVES (Model missed these):")
        for item in missed:
            lines.append(f"  -> '{item}'")

    hallucinated = [p for p in pred_texts if p not in true_texts]
    if hallucinated:
        lines.append("FALSE POSITIVES (Model should not have highlighted this):")
        for item in hallucinated:
            lines.append(f"  -> '{item}'")

    lines.append("=" * 80)
    return "\n".join(lines)


def compare_predictions(predictions, gold_items, source_name, writer):
    prediction_map = {get_item_key(item): item for item in predictions}

    total = 0
    mismatches = 0
    missing_predictions = 0

    writer.write(f"\nComparing against {source_name}\n")

    for gold_item in gold_items:
        key = get_item_key(gold_item)
        pred_item = prediction_map.get(key)
        total += 1

        if pred_item is None:
            missing_predictions += 1
            writer.write("\n" + "=" * 80 + "\n")
            writer.write("MISSING PREDICTION\n")
            writer.write("-" * 80 + "\n")
            writer.write(
                f"text_id={gold_item['text_id']}, topic_id={gold_item['topic_id']}, "
                f"cluster_id={gold_item['cluster_id']}"
            )
            writer.write("\n" + "=" * 80 + "\n\n")
            continue

        report = format_error_analysis(
            gold_item["text"],
            normalize_annotations(gold_item),
            normalize_annotations(pred_item),
            metadata={
                "text_id": gold_item["text_id"],
                "topic_id": gold_item["topic_id"],
                "cluster_id": gold_item["cluster_id"],
                "topic_name": gold_item["topic_name"],
            },
        )
        if report is not None:
            mismatches += 1
            writer.write(report + "\n")

    summary = (
        f"Summary for {source_name}: total={total}, mismatches={mismatches}, "
        f"missing_predictions={missing_predictions}\n"
    )
    writer.write(summary)
    return summary.strip()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-path", type=Path, required=True)
    parser.add_argument(
        "--gold-paths",
        type=Path,
        nargs="+",
        default=[
            Path("dataset/test-dataset/0.jsonl"),
            Path("dataset/test-dataset/3.jsonl"),
            Path("dataset/test-dataset/4.jsonl"),
            Path("dataset/test-dataset/5.jsonl"),
            Path("dataset/test-dataset/6.jsonl"),
            Path("dataset/test-dataset/7.jsonl"),
        ],
        help="One or more annotator JSONL files, e.g. dataset/test-dataset/0.jsonl",
    )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=20,
        help="Randomly compare only this many items from each annotator file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed used when --sample-size is set.",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("output/error_analysis_report.txt"),
        help="Optional text file path for a readable mismatch report.",
    )
    args = parser.parse_args()

    predictions = load_jsonl(args.predictions_path)
    rng = random.Random(args.seed)

    if args.output_path is not None:
        args.output_path.parent.mkdir(parents=True, exist_ok=True)
        writer = open(args.output_path, "w", encoding="utf-8")
    else:
        writer = None

    summaries = []

    try:
        output = writer if writer is not None else sys.stdout

        if args.sample_size is not None:
            output.write(
                f"Using random sample of {args.sample_size} items per annotator file "
                f"(seed={args.seed})\n"
            )

        for gold_path in args.gold_paths:
            gold_items = load_jsonl(gold_path)
            if args.sample_size is not None and len(gold_items) > args.sample_size:
                gold_items = rng.sample(gold_items, args.sample_size)
            summaries.append(
                compare_predictions(predictions, gold_items, gold_path.name, output)
            )
    finally:
        if writer is not None:
            writer.close()

    for summary in summaries:
        print(summary)
    if args.output_path is not None:
        print(f"Detailed report saved to {args.output_path}")


if __name__ == "__main__":
    main()
