import argparse
import json
from pathlib import Path
import logging
from typing import Sequence

import numpy as np
import pandas as pd

from common import load_jsonl, calculate_pair_metrics, align_data, topic_scores_fixed_rater1_vs_group, bootstrap_topic_macro_ci_df, derive_text_presence_confusion


logger = logging.getLogger(__name__)


P1_ID = 0
HUMAN_IDS = [3, 4, 5, 6, 7]
TMP_ID = 9999

def load_gt(path: Path) -> dict[int, list[dict]]:
    data = {}

    p1_path = path / "test-dataset" / f"{P1_ID}.jsonl"
    if not p1_path.exists():
        raise FileNotFoundError(f"Ground truth file for P1 not found at {p1_path}")

    data[P1_ID] = load_jsonl(p1_path)

    for human_id in HUMAN_IDS:
        human_path = path / "test-dataset" / f"{human_id}.jsonl"
        if not human_path.exists():
            raise FileNotFoundError(f"Ground truth file for human rater {human_id} not found at {human_path}")
        data[human_id] = load_jsonl(human_path)

    return data


def create_df(
    pred: list[dict],
    gt: dict[int, list[dict]],
) -> pd.DataFrame:
    dfs = []
    for rater_id, rater_data in gt.items():
        logger.info(f"Calculating metrics for rater {rater_id} ...")
        gt_aligned, pred_aligned = align_data(rater_data, pred)
        df = calculate_pair_metrics(
            predictions=pred_aligned,
            ground_truth=gt_aligned,
            rater_1=str(TMP_ID),
            rater_2=str(rater_id),
            against_p1=(rater_id == P1_ID),
        )
        logger.info(f"Metrics for rater {rater_id}: {df}")
        dfs.append(df)

    df = pd.concat(dfs, ignore_index=True)
    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Cross Encoder model for topic tagging.")
    parser.add_argument("--pred", type=Path, required=True, help="Path to the predictions JSONL file.")
    parser.add_argument("--gt", type=Path, required=True, help="Path to the directory containing the CzechTopic dataset.")

    parser.add_argument("--n-boot", type=int, default=20000, help="Number of bootstrap samples for confidence interval estimation.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for bootstrap sampling.")

    parser.add_argument("--save-path", type=Path, default=None, help="Path to save detailed evaluation results as CSV (sep=\t).")

    return parser.parse_args()


def main():
    args = parse_args()

    logger.info(f"Loading predictions from {args.pred} ...")
    with args.pred.open("r") as f:
        pred = [json.loads(line) for line in f]
    logger.info(f"Loaded {len(pred)} samples from predictions data.")

    logger.info(f"Loading ground truth data from {args.gt} ...")
    gt = load_gt(args.gt)
    logger.info("Loaded ground truth data for P1 and human raters.")

    logger.info("Creating evaluation DataFrame ...")
    df = create_df(pred, gt)
    logger.info("Evaluation DataFrame created.")

    if args.save_path is not None:
        logger.info(f"Saving intermediate evaluation DataFrame to {args.save_path} ...")
        df.to_csv(args.save_path, index=False, sep="\t")
        logger.info("Intermediate evaluation DataFrame saved.")

    logger.info("Deriving text presence confusion ...")
    df = derive_text_presence_confusion(df)
    logger.info("Text presence confusion derived.")

    logger.info("Calculating topic-level scores and bootstrap confidence intervals against humans ...")
    df_vs_humans = df[df["rater_2_id"].astype(int).isin(HUMAN_IDS)].copy()
    topic_scores_m = topic_scores_fixed_rater1_vs_group(
        df=df_vs_humans,
        compare_col="rater_2_id",
    )
    result = bootstrap_topic_macro_ci_df(
        topic_scores=topic_scores_m,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    logger.info("Topic-level scores and bootstrap confidence intervals against humans calculated.")

    print("Evaluation results against human raters:")
    print(result)

    logger.info("Calculating topic-level scores and bootstrap confidence intervals against P1 ...")
    df_vs_p1 = df[df["rater_2_id"].astype(int) == P1_ID].copy()
    topic_scores_m_p1 = topic_scores_fixed_rater1_vs_group(
        df=df_vs_p1,
        compare_col="rater_2_id",
    )
    result_p1 = bootstrap_topic_macro_ci_df(
        topic_scores=topic_scores_m_p1,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    logger.info("Topic-level scores and bootstrap confidence intervals against P1 calculated.")

    print("\nEvaluation results against P1:")
    print(result_p1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
