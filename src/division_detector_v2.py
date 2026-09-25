from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree


SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)

EVENT_COLUMNS = [
    "volume",
    "parent_track_id",
    "parent_t",
    "parent_z",
    "parent_y",
    "parent_x",
    "daughter_1_track_id",
    "daughter_1_t",
    "daughter_1_z",
    "daughter_1_y",
    "daughter_1_x",
    "daughter_2_track_id",
    "daughter_2_t",
    "daughter_2_z",
    "daughter_2_y",
    "daughter_2_x",
    "confidence",
    "parent_child_dist_um_mean",
    "parent_child_dist_um_max",
    "daughter_separation_um",
    "parent_age_frames",
    "daughter_1_persistence_frames",
    "daughter_2_persistence_frames",
]

EDGE_COLUMNS = [
    "volume",
    "source_track_id",
    "source_t",
    "target_track_id",
    "target_t",
    "distance_um",
    "edge_type",
    "confidence",
]


@dataclass(frozen=True)
class DivisionDetectorConfig:
    """GT-informed defaults from gt_divisions.csv."""

    max_parent_child_dist_um: float = 8.0
    min_daughter_separation_um: float = 8.0
    min_parent_age_frames: int = 10
    min_daughter_persistence: int = 2
    max_gap: int = 1
    require_new_daughter_track: bool = True


def _validate_tracks(tracks_df: pd.DataFrame) -> None:
    required = {"volume", "track_id", "t", "z", "y", "x"}
    missing = sorted(required - set(tracks_df.columns))
    if missing:
        raise ValueError(f"tracks_df missing required columns: {missing}")


def _coords_um(frame: pd.DataFrame) -> np.ndarray:
    return frame[["z", "y", "x"]].to_numpy(dtype=np.float64) * SCALE


def _track_meta(tracks_df: pd.DataFrame) -> pd.DataFrame:
    meta = (
        tracks_df.groupby(["volume", "track_id"], sort=False)["t"]
        .agg(start_t="min", end_t="max", length="count")
        .reset_index()
    )
    return meta


def _build_frame_lookup(tracks_df: pd.DataFrame) -> dict[tuple[str, int], dict]:
    lookup = {}
    for (volume, t), frame in tracks_df.groupby(["volume", "t"], sort=False):
        frame = frame.reset_index(drop=True)
        coords = _coords_um(frame)
        lookup[(str(volume), int(t))] = {
            "frame": frame,
            "coords": coords,
            "tree": cKDTree(coords) if len(coords) else None,
        }
    return lookup


def _persistence_frames(
    tracks_df: pd.DataFrame,
    volume: str,
    track_id: int,
    start_t: int,
    min_frames: int,
) -> int:
    track = tracks_df[
        (tracks_df["volume"] == volume)
        & (tracks_df["track_id"] == track_id)
        & (tracks_df["t"] >= start_t)
        & (tracks_df["t"] < start_t + min_frames)
    ]
    return int(track["t"].nunique())


def _score_pair(
    parent_child_distances: tuple[float, float],
    daughter_separation_um: float,
    parent_age_frames: int,
    daughter_persistence: tuple[int, int],
    config: DivisionDetectorConfig,
) -> float:
    mean_parent_child = float(np.mean(parent_child_distances))
    distance_score = 1.0 - min(mean_parent_child / config.max_parent_child_dist_um, 1.0)
    separation_target_um = 10.53
    separation_score = 1.0 - min(abs(daughter_separation_um - separation_target_um) / separation_target_um, 1.0)
    age_score = min(parent_age_frames / 16.0, 1.0)
    persistence_score = min(min(daughter_persistence) / max(config.min_daughter_persistence, 1), 1.0)
    return float(
        0.35 * distance_score
        + 0.30 * separation_score
        + 0.20 * age_score
        + 0.15 * persistence_score
    )


def detect_divisions(
    tracks_df: pd.DataFrame,
    max_parent_child_dist_um: float = 8.0,
    min_daughter_separation_um: float = 8.0,
    min_parent_age_frames: int = 10,
    min_daughter_persistence: int = 2,
    max_gap: int = 1,
    require_new_daughter_track: bool = True,
) -> pd.DataFrame:
    """
    Detect cell division events from linked tracks.

    Defaults are based on train GT:
    parent-child distance mean 5.81 um, daughter separation mean 10.53 um,
    parent lineage age median 16 frames, and daughter persistence typically >2 frames.
    """
    _validate_tracks(tracks_df)
    if tracks_df.empty:
        return pd.DataFrame(columns=EVENT_COLUMNS)

    config = DivisionDetectorConfig(
        max_parent_child_dist_um=max_parent_child_dist_um,
        min_daughter_separation_um=min_daughter_separation_um,
        min_parent_age_frames=min_parent_age_frames,
        min_daughter_persistence=min_daughter_persistence,
        max_gap=max_gap,
        require_new_daughter_track=require_new_daughter_track,
    )

    tracks = tracks_df.copy()
    tracks["volume"] = tracks["volume"].astype(str)
    tracks["track_id"] = tracks["track_id"].astype(np.int64)
    tracks["t"] = tracks["t"].astype(np.int64)
    tracks = tracks.sort_values(["volume", "track_id", "t"]).reset_index(drop=True)

    meta = _track_meta(tracks)
    start_lookup = {
        (str(row.volume), int(row.track_id)): int(row.start_t)
        for row in meta.itertuples(index=False)
    }
    frame_lookup = _build_frame_lookup(tracks)

    events = []
    emitted = set()

    for (volume, parent_track_id), parent_track in tracks.groupby(["volume", "track_id"], sort=True):
        parent_track = parent_track.sort_values("t").reset_index(drop=True)
        if len(parent_track) < config.min_parent_age_frames:
            continue

        for parent_idx, parent in parent_track.iterrows():
            parent_t = int(parent.t)
            parent_age_frames = int(parent_idx) + 1
            if parent_age_frames < config.min_parent_age_frames:
                continue

            parent_coord_um = parent[["z", "y", "x"]].to_numpy(dtype=np.float64) * SCALE
            candidates = []

            for gap in range(1, config.max_gap + 1):
                daughter_t = parent_t + gap
                frame_data = frame_lookup.get((str(volume), daughter_t))
                if frame_data is None or frame_data["tree"] is None:
                    continue

                nearby_indices = frame_data["tree"].query_ball_point(
                    parent_coord_um,
                    config.max_parent_child_dist_um,
                )
                frame = frame_data["frame"]
                coords = frame_data["coords"]

                for idx in nearby_indices:
                    daughter = frame.iloc[int(idx)]
                    daughter_track_id = int(daughter.track_id)
                    start_t = start_lookup.get((str(volume), daughter_track_id))
                    is_new_track = start_t == daughter_t
                    persistence = _persistence_frames(
                        tracks,
                        str(volume),
                        daughter_track_id,
                        daughter_t,
                        config.min_daughter_persistence,
                    )
                    if persistence < config.min_daughter_persistence:
                        continue

                    candidates.append(
                        {
                            "row": daughter,
                            "track_id": daughter_track_id,
                            "t": daughter_t,
                            "coord_um": coords[int(idx)],
                            "distance_um": float(np.linalg.norm(coords[int(idx)] - parent_coord_um)),
                            "is_new_track": bool(is_new_track),
                            "persistence": persistence,
                        }
                    )

            if len(candidates) < 2:
                continue

            by_track = {}
            for candidate in sorted(candidates, key=lambda item: item["distance_um"]):
                by_track.setdefault(candidate["track_id"], candidate)
            candidates = list(by_track.values())

            best_pair = None
            best_score = -np.inf
            for i in range(len(candidates)):
                for j in range(i + 1, len(candidates)):
                    a = candidates[i]
                    b = candidates[j]
                    if a["track_id"] == b["track_id"]:
                        continue
                    if config.require_new_daughter_track and not (a["is_new_track"] or b["is_new_track"]):
                        continue

                    daughter_separation_um = float(np.linalg.norm(a["coord_um"] - b["coord_um"]))
                    if daughter_separation_um < config.min_daughter_separation_um:
                        continue

                    score = _score_pair(
                        (a["distance_um"], b["distance_um"]),
                        daughter_separation_um,
                        parent_age_frames,
                        (a["persistence"], b["persistence"]),
                        config,
                    )
                    if score > best_score:
                        best_score = score
                        best_pair = (a, b, daughter_separation_um)

            if best_pair is None:
                continue

            daughter_1, daughter_2, daughter_separation_um = best_pair
            key = (str(volume), int(parent_track_id), parent_t)
            if key in emitted:
                continue
            emitted.add(key)

            daughter_rows = sorted(
                [daughter_1, daughter_2],
                key=lambda item: (int(item["t"]), int(item["track_id"])),
            )
            d1, d2 = daughter_rows
            d1_row = d1["row"]
            d2_row = d2["row"]
            events.append(
                {
                    "volume": str(volume),
                    "parent_track_id": int(parent_track_id),
                    "parent_t": parent_t,
                    "parent_z": float(parent.z),
                    "parent_y": float(parent.y),
                    "parent_x": float(parent.x),
                    "daughter_1_track_id": int(d1["track_id"]),
                    "daughter_1_t": int(d1["t"]),
                    "daughter_1_z": float(d1_row.z),
                    "daughter_1_y": float(d1_row.y),
                    "daughter_1_x": float(d1_row.x),
                    "daughter_2_track_id": int(d2["track_id"]),
                    "daughter_2_t": int(d2["t"]),
                    "daughter_2_z": float(d2_row.z),
                    "daughter_2_y": float(d2_row.y),
                    "daughter_2_x": float(d2_row.x),
                    "confidence": best_score,
                    "parent_child_dist_um_mean": float(np.mean([d1["distance_um"], d2["distance_um"]])),
                    "parent_child_dist_um_max": float(max(d1["distance_um"], d2["distance_um"])),
                    "daughter_separation_um": daughter_separation_um,
                    "parent_age_frames": parent_age_frames,
                    "daughter_1_persistence_frames": int(d1["persistence"]),
                    "daughter_2_persistence_frames": int(d2["persistence"]),
                }
            )

    return pd.DataFrame(events, columns=EVENT_COLUMNS)


def division_events_to_edges(divisions_df: pd.DataFrame) -> pd.DataFrame:
    if divisions_df.empty:
        return pd.DataFrame(columns=EDGE_COLUMNS)

    rows = []
    for event in divisions_df.itertuples(index=False):
        distances = [
            float(event.parent_child_dist_um_mean),
            float(event.parent_child_dist_um_mean),
        ]
        daughters = [
            (int(event.daughter_1_track_id), int(event.daughter_1_t), distances[0]),
            (int(event.daughter_2_track_id), int(event.daughter_2_t), distances[1]),
        ]
        for target_track_id, target_t, distance_um in daughters:
            rows.append(
                {
                    "volume": str(event.volume),
                    "source_track_id": int(event.parent_track_id),
                    "source_t": int(event.parent_t),
                    "target_track_id": target_track_id,
                    "target_t": target_t,
                    "distance_um": distance_um,
                    "edge_type": "division",
                    "confidence": float(event.confidence),
                }
            )
    return pd.DataFrame(rows, columns=EDGE_COLUMNS)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tracks", required=True)
    parser.add_argument("--output", default="divisions_detected.csv")
    parser.add_argument("--edges-output", default=None)
    parser.add_argument("--max-parent-child-dist-um", type=float, default=8.0)
    parser.add_argument("--min-daughter-separation-um", type=float, default=8.0)
    parser.add_argument("--min-parent-age-frames", type=int, default=10)
    parser.add_argument("--min-daughter-persistence", type=int, default=2)
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--allow-existing-daughter-tracks", action="store_true")
    args = parser.parse_args()

    tracks = pd.read_csv(args.tracks)
    divisions = detect_divisions(
        tracks,
        max_parent_child_dist_um=args.max_parent_child_dist_um,
        min_daughter_separation_um=args.min_daughter_separation_um,
        min_parent_age_frames=args.min_parent_age_frames,
        min_daughter_persistence=args.min_daughter_persistence,
        max_gap=args.max_gap,
        require_new_daughter_track=not args.allow_existing_daughter_tracks,
    )
    divisions.to_csv(args.output, index=False)

    print(f"Detected division events: {len(divisions)}")
    print(f"Saved division events to {args.output}")
    if not divisions.empty:
        print(
            "Median parent-child distance: "
            f"{divisions['parent_child_dist_um_mean'].median():.3f} um"
        )
        print(
            "Median daughter separation: "
            f"{divisions['daughter_separation_um'].median():.3f} um"
        )
        print(
            "Median parent age: "
            f"{divisions['parent_age_frames'].median():.1f} frames"
        )

    if args.edges_output:
        edges = division_events_to_edges(divisions)
        edges.to_csv(args.edges_output, index=False)
        print(f"Saved division edges to {args.edges_output}")
        print(f"Division edges: {len(edges)}")


if __name__ == "__main__":
    main()
