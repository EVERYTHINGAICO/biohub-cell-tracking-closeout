from __future__ import annotations

import argparse
from pathlib import Path

from metric_local import evaluate_submission


DEFAULT_TRAIN_DIR = Path(
    "/mnt/d/descargas al dico vergas/biohub-cell-tracking-during-development/train"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Kaggle submission CSV against local train GEFF ground truth."
    )
    parser.add_argument(
        "--submission",
        default="submission_dog.csv",
        help="Submission CSV to validate.",
    )
    parser.add_argument(
        "--train-dir",
        default=str(DEFAULT_TRAIN_DIR),
        help="Directory containing train .geff files.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset ids to evaluate. Defaults to datasets present in the CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    metrics, summary = evaluate_submission(
        submission_csv=args.submission,
        train_dir=args.train_dir,
        datasets=args.datasets,
    )

    print("Per-dataset metrics")
    print(
        "dataset,pred_nodes,gt_nodes,estimated_nodes,matched_nodes,"
        "edge_tp,edge_fp,edge_fn,edge_jaccard,adjusted_edge_jaccard,"
        "division_tp,division_fp,division_fn,division_jaccard,"
        "official_like_score,requested_combined_score"
    )
    for row in metrics:
        print(
            f"{row.dataset},{row.pred_nodes},{row.gt_nodes},{row.estimated_nodes},"
            f"{row.matched_nodes},{row.edge_counts.tp},{row.edge_counts.fp},"
            f"{row.edge_counts.fn},{row.edge_jaccard:.6f},"
            f"{row.adjusted_edge_jaccard:.6f},{row.division_counts.tp},"
            f"{row.division_counts.fp},{row.division_counts.fn},"
            f"{row.division_jaccard:.6f},{row.official_like_score:.6f},"
            f"{row.requested_combined_score:.6f}"
        )

    print()
    print("Summary")
    for key in [
        "datasets",
        "pred_nodes",
        "gt_nodes",
        "estimated_nodes",
        "matched_nodes",
        "edge_tp",
        "edge_fp",
        "edge_fn",
        "edge_jaccard",
        "adjusted_edge_jaccard",
        "requested_adjusted_edge_jaccard",
        "division_tp",
        "division_fp",
        "division_fn",
        "division_jaccard",
        "official_like_score",
        "requested_combined_score",
    ]:
        value = summary[key]
        if isinstance(value, float):
            print(f"{key}: {value:.6f}")
        else:
            print(f"{key}: {value}")

    print()
    print("Comparable-to-Kaggle local score: " f"{summary['official_like_score']:.6f}")


if __name__ == "__main__":
    main()
