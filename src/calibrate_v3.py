from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from detection_dog_v2 import DEFAULT_DOG_SCALES, SCALE, detect_volume
from format_submission import format_submission
from metric_local_v2 import compute_metrics


MAX_TRIALS = 30
DEFAULT_SEED = 20260702


def evenly_spaced(items: list[Path], n: int) -> list[Path]:
    if n <= 0:
        return []
    if n >= len(items):
        return list(items)
    indices = np.linspace(0, len(items) - 1, n, dtype=int).tolist()
    selected = []
    seen = set()
    for idx in indices:
        if idx not in seen:
            selected.append(items[idx])
            seen.add(idx)
    for item in items:
        if len(selected) >= n:
            break
        if item not in selected:
            selected.append(item)
    return selected[:n]


def select_validation_subset(train_dir: Path, subset_size: int = 15) -> list[Path]:
    zarr_paths = sorted(train_dir.glob("*.zarr"))
    if not zarr_paths:
        raise FileNotFoundError(f"No .zarr volumes found in {train_dir}")

    by_prefix: dict[str, list[Path]] = {}
    for path in zarr_paths:
        prefix = path.stem.split("_", 1)[0]
        by_prefix.setdefault(prefix, []).append(path)

    prefixes = sorted(by_prefix)
    if len(prefixes) < 2:
        return evenly_spaced(zarr_paths, min(subset_size, len(zarr_paths)))

    first_n = (subset_size + 1) // 2
    second_n = subset_size - first_n
    selected = []
    selected.extend(evenly_spaced(by_prefix[prefixes[0]], first_n))
    selected.extend(evenly_spaced(by_prefix[prefixes[1]], second_n))
    return sorted(selected)[:subset_size]


def filter_topk_per_frame(detections: pd.DataFrame, topk: int | None) -> pd.DataFrame:
    if topk is None or topk <= 0:
        return detections
    if detections.empty:
        return detections
    return (
        detections.groupby(["volume", "t"], group_keys=False)
        .apply(lambda frame: frame.nlargest(topk, "score"))
        .reset_index(drop=True)
    )


def track_cells_physical(
    detections: pd.DataFrame,
    max_link_um: float,
    min_track_length: int = 2,
) -> pd.DataFrame:
    rows = []
    next_track_id = 0

    for volume, volume_df in detections.groupby("volume", sort=True):
        active: dict[int, np.ndarray] = {}
        active_rows: dict[int, dict] = {}

        for t in sorted(volume_df["t"].unique()):
            frame = volume_df[volume_df["t"] == t].reset_index(drop=True)
            positions = frame[["z", "y", "x"]].to_numpy(dtype=np.float64)
            positions_um = positions * SCALE
            scores = frame["score"].to_numpy(dtype=np.float64)

            if len(active) == 0:
                for pos, score in zip(positions, scores):
                    next_track_id += 1
                    row = {
                        "volume": volume,
                        "t": int(t),
                        "z": float(pos[0]),
                        "y": float(pos[1]),
                        "x": float(pos[2]),
                        "score": float(score),
                        "track_id": next_track_id,
                    }
                    rows.append(row)
                    active[next_track_id] = pos * SCALE
                    active_rows[next_track_id] = row
                continue

            track_ids = list(active.keys())
            active_pos_um = np.vstack([active[track_id] for track_id in track_ids])
            cost = cdist(active_pos_um, positions_um)
            cost[cost > max_link_um] = 1e9

            row_idx, col_idx = linear_sum_assignment(cost)
            matched_tracks = set()
            matched_detections = set()
            new_active: dict[int, np.ndarray] = {}
            new_active_rows: dict[int, dict] = {}

            for r, c in zip(row_idx, col_idx):
                if cost[r, c] >= 1e9:
                    continue
                track_id = track_ids[r]
                pos = positions[c]
                row = {
                    "volume": volume,
                    "t": int(t),
                    "z": float(pos[0]),
                    "y": float(pos[1]),
                    "x": float(pos[2]),
                    "score": float(scores[c]),
                    "track_id": track_id,
                }
                rows.append(row)
                new_active[track_id] = positions_um[c]
                new_active_rows[track_id] = row
                matched_tracks.add(track_id)
                matched_detections.add(int(c))

            for det_idx, pos in enumerate(positions):
                if det_idx in matched_detections:
                    continue
                next_track_id += 1
                row = {
                    "volume": volume,
                    "t": int(t),
                    "z": float(pos[0]),
                    "y": float(pos[1]),
                    "x": float(pos[2]),
                    "score": float(scores[det_idx]),
                    "track_id": next_track_id,
                }
                rows.append(row)
                new_active[next_track_id] = positions_um[det_idx]
                new_active_rows[next_track_id] = row

            active = new_active
            active_rows = new_active_rows

    tracks = pd.DataFrame(rows)
    if tracks.empty or min_track_length <= 1:
        return tracks

    lengths = tracks.groupby(["volume", "track_id"]).size()
    keep = lengths[lengths >= min_track_length].index
    keep_index = pd.MultiIndex.from_frame(tracks[["volume", "track_id"]])
    return tracks[keep_index.isin(keep)].reset_index(drop=True)


def run_trial(
    trial: optuna.Trial,
    train_dir: Path,
    subset_paths: list[Path],
    work_dir: Path,
    max_peaks_per_frame: int,
    topk_per_frame: int,
    min_track_length: int,
) -> float:
    rel_threshold = trial.suggest_float("rel_threshold", 0.03, 0.06)
    min_distance_um = trial.suggest_float("min_distance_um", 2.5, 4.0)
    max_link_um = trial.suggest_float("max_link_um", 6.0, 10.0)

    trial_dir = work_dir / f"trial_{trial.number:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    all_detections = []
    for i, zarr_path in enumerate(subset_paths, start=1):
        print(
            f"[trial {trial.number:03d}] detecting {i}/{len(subset_paths)} {zarr_path.name}",
            flush=True,
        )
        detections = detect_volume(
            zarr_path=zarr_path,
            xy_downsample=4,
            dog_scales=DEFAULT_DOG_SCALES,
            rel_threshold=rel_threshold,
            abs_percentile=50.0,
            min_distance_um=min_distance_um,
            max_peaks=max_peaks_per_frame,
            refine=True,
        )
        all_detections.append(detections)

    detections_df = pd.concat(all_detections, ignore_index=True)
    detections_df = filter_topk_per_frame(detections_df, topk_per_frame)
    detections_path = trial_dir / "detections.csv"
    detections_df.to_csv(detections_path, index=False)

    tracks_df = track_cells_physical(
        detections_df,
        max_link_um=max_link_um,
        min_track_length=min_track_length,
    )
    tracks_path = trial_dir / "tracks.csv"
    tracks_df.to_csv(tracks_path, index=False)

    submission_path = trial_dir / "submission.csv"
    format_submission(tracks_df, submission_path)
    submission_df = pd.read_csv(submission_path)
    metrics = compute_metrics(submission_df, train_dir)

    metrics_json = {
        key: value
        for key, value in metrics.items()
        if key != "per_volume"
    }
    metrics_json.update(
        {
            "rel_threshold": rel_threshold,
            "min_distance_um": min_distance_um,
            "max_link_um": max_link_um,
            "detections": int(len(detections_df)),
            "tracks": int(len(tracks_df)),
            "subset_volumes": [path.stem for path in subset_paths],
        }
    )
    with (trial_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics_json, f, indent=2)

    for key, value in metrics_json.items():
        if isinstance(value, (int, float, str)):
            trial.set_user_attr(key, value)

    score = float(metrics["combined_score"])
    print(
        f"[trial {trial.number:03d}] score={score:.6f} "
        f"edge={metrics['edge_jaccard']:.6f} div={metrics['division_jaccard']:.6f} "
        f"pred_nodes={metrics['pred_nodes']}",
        flush=True,
    )
    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Optuna calibration on a deterministic 15-volume train subset."
    )
    parser.add_argument(
        "--train-dir",
        default="/mnt/d/descargas al dico vergas/biohub-cell-tracking-during-development/train",
    )
    parser.add_argument("--n-trials", type=int, default=30)
    parser.add_argument("--subset-size", type=int, default=15)
    parser.add_argument("--output", default="calibration_results_v3.csv")
    parser.add_argument("--work-dir", default="artifacts/calibration_v3")
    parser.add_argument("--max-peaks-per-frame", type=int, default=500)
    parser.add_argument("--topk-per-frame", type=int, default=500)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.n_trials > MAX_TRIALS:
        print(f"Requested {args.n_trials} trials; clamping to max {MAX_TRIALS}.")
        args.n_trials = MAX_TRIALS

    train_dir = Path(args.train_dir)
    work_dir = Path(args.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    subset_paths = select_validation_subset(train_dir, subset_size=args.subset_size)

    print("Validation subset:")
    for path in subset_paths:
        print(f"  {path.stem}")

    sampler = optuna.samplers.TPESampler(seed=args.seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: run_trial(
            trial=trial,
            train_dir=train_dir,
            subset_paths=subset_paths,
            work_dir=work_dir,
            max_peaks_per_frame=args.max_peaks_per_frame,
            topk_per_frame=args.topk_per_frame,
            min_track_length=args.min_track_length,
        ),
        n_trials=args.n_trials,
    )

    trials = study.trials_dataframe().sort_values("value", ascending=False)
    trials.to_csv(args.output, index=False)

    print("\n=== OPTIMIZATION COMPLETE ===")
    print(f"Best score: {study.best_value:.6f}")
    print(f"Best params: {study.best_params}")
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
