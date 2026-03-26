""" Evaluation script for evaluating a set topic localization results against a second set of results. 

Author: Martin Kostelník (ikostelnik@fit.vut.cz)
Date: 11.3.2026
"""

import argparse
from pathlib import Path
import logging

from common import load_jsonl, calculate_pair_metrics, align_data, topic_scores_fixed_rater1_vs_group, bootstrap_topic_macro_ci_df, derive_text_presence_confusion


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Cross Encoder model for topic tagging.")
    parser.add_argument("--pred", type=Path, required=True, help="Path to the predictions JSONL file.")
    parser.add_argument("--gt", type=Path, required=True, help="Path to the ground truth data JSONL file.")
    parser.add_argument("--save-path", type=Path, default=None, help="Path to save detailed evaluation results as CSV (sep=\t).")
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading ground truth data from {args.gt} ...")
    gt = load_jsonl(args.gt)
    logger.info(f"Loaded {len(gt)} samples from ground truth data.")

    logger.info(f"Loading predictions from {args.pred} ...")
    pred = load_jsonl(args.pred)
    logger.info(f"Loaded {len(pred)} samples from predictions data.")

    logger.info("Filtering samples that are not present in both ground truth and predictions ...")
    gt, pred = align_data(gt, pred)
    logger.info(f"After filtering to common keys, {len(gt)} samples remain in ground truth and {len(pred)} samples remain in predictions.")

    assert len(pred) == len(gt), "Number of predictions and ground truth samples must be the same"

    logger.info("Calculating pair metrics ...")
    df = calculate_pair_metrics(
        predictions=pred,
        ground_truth=gt,
    )
    logger.info("Pair metrics calculation completed.")

    if args.save_path is not None:
        logger.info(f"Saving detailed evaluation results to {args.save_path} ...")
        df.to_csv(args.save_path, index=False, sep="\t")
        logger.info("Detailed evaluation results saved.")

    logger.info("Deriving text presence confusion ...")
    df = derive_text_presence_confusion(df)
    logger.info("Text presence confusion derived.")

    logger.info("Calculating topic-level scores and bootstrap confidence intervals ...")
    topic_scores_m = topic_scores_fixed_rater1_vs_group(
        df=df,
        compare_col="rater_2_id",
    )
    result = bootstrap_topic_macro_ci_df(
        topic_scores=topic_scores_m,
        n_boot=20000,
        seed=42,
    )
    logging.info("Topic-level scores and bootstrap confidence intervals calculated.")
    
    print(result)
    logger.info("Done.")


if __name__ == "__main__":
    main()
