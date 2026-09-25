from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial import cKDTree

from tracker import track_cells


PHYSICAL_SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


def _coords_um(frame: pd.DataFrame) -> np.ndarray:
    return frame[["z", "y", "x"]].to_numpy(dtype=np.float64) * PHYSICAL_SCALE


def detect_divisions(
    tracks_df: pd.DataFrame,
    max_gap: int = 1,
    max_distance_um: float = 8.0,
    max_children: int = 2,
    require_new_track: bool = True,
) -> pd.DataFrame:
    required = {"volume", "track_id", "t", "z", "y", "x"}
    missing = sorted(required - set(tracks_df.columns))
    if missing:
        raise ValueError(f"tracks_df missing required columns: {missing}")

    tracks = tracks_df.copy()
    tracks["track_id"] = tracks["track_id"].astype(np.int64)
    tracks["t"] = tracks["t"].astype(np.int64)
    tracks = tracks.sort_values(["volume", "t", "track_id"]).reset_index(drop=True)

    track_start = tracks.groupby(["volume", "track_id"])["t"].min().to_dict()
    frame_lookup = {}
    for (volume, t), frame in tracks.groupby(["volume", "t"], sort=False):
        frame = frame.reset_index(drop=True)
        coords = _coords_um(frame)
        frame_lookup[(volume, int(t))] = {
            "frame": frame,
            "coords": coords,
            "tree": cKDTree(coords) if len(coords) else None,
        }

    division_edges = []
    emitted = set()

    for _, parent in tracks.iterrows():
        volume = parent["volume"]
        parent_track = int(parent["track_id"])
        parent_t = int(parent["t"])

        candidates = []
        for gap in range(1, max_gap + 1):
            child_t = parent_t + gap
            frame_data = frame_lookup.get((volume, child_t))
            if frame_data is None or frame_data["tree"] is None:
                continue

            parent_coord = parent[["z", "y", "x"]].to_numpy(dtype=np.float64) * PHYSICAL_SCALE
            child_indices = frame_data["tree"].query_ball_point(parent_coord, max_distance_um)
            if not child_indices:
                continue

            frame = frame_data["frame"]
            coords = frame_data["coords"]
            for child_idx in child_indices:
                child = frame.iloc[int(child_idx)]
                child_track = int(child.track_id)
                if child_track == parent_track:
                    child_kind = "continuation"
                elif track_start.get((volume, child_track)) == child_t:
                    child_kind = "new_track"
                else:
                    child_kind = "existing_track"

                if require_new_track and child_kind == "existing_track":
                    continue

                candidates.append(
                    {
                        "volume": volume,
                        "source_track_id": parent_track,
                        "source_t": parent_t,
                        "target_track_id": child_track,
                        "target_t": child_t,
                        "distance_um": float(np.linalg.norm(coords[int(child_idx)] - parent_coord)),
                        "child_kind": child_kind,
                    }
                )

        if len(candidates) < 2:
            continue
        if require_new_track and not any(c["child_kind"] == "new_track" for c in candidates):
            continue

        continuation = [
            c for c in candidates
            if c["target_track_id"] == parent_track and c["child_kind"] == "continuation"
        ]
        new_tracks = [
            c for c in candidates
            if c["target_track_id"] != parent_track and c["child_kind"] == "new_track"
        ]

        selected = []
        if continuation:
            selected.append(min(continuation, key=lambda c: c["distance_um"]))
        selected.extend(sorted(new_tracks, key=lambda c: c["distance_um"]))

        if len(selected) < 2:
            selected = sorted(candidates, key=lambda c: c["distance_um"])

        selected = selected[:max_children]
        if len({c["target_track_id"] for c in selected}) < 2:
            continue

        for edge in selected:
            key = (
                edge["volume"],
                edge["source_track_id"],
                edge["source_t"],
                edge["target_track_id"],
                edge["target_t"],
            )
            if key in emitted:
                continue
            emitted.add(key)
            edge = dict(edge)
            edge["edge_type"] = "division"
            division_edges.append(edge)

    return pd.DataFrame(
        division_edges,
        columns=[
            "volume",
            "source_track_id",
            "source_t",
            "target_track_id",
            "target_t",
            "distance_um",
            "child_kind",
            "edge_type",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, default=None)
    parser.add_argument("--input-tracks", type=str, default=None)
    parser.add_argument("--output", type=str, default="tracks_v2.csv")
    parser.add_argument("--division-edges", type=str, default="divisions_v2.csv")
    parser.add_argument("--max-distance", type=float, default=50.0)
    parser.add_argument("--division-distance-um", type=float, default=8.0)
    parser.add_argument("--max-gap", type=int, default=1)
    args = parser.parse_args()

    if args.input_tracks:
        tracks_df = pd.read_csv(args.input_tracks)
        print(f"Loaded {len(tracks_df)} track points from {args.input_tracks}")
    elif args.predictions:
        predictions_df = pd.read_csv(args.predictions)
        print(f"Loaded {len(predictions_df)} detections")
        if predictions_df.empty:
            raise ValueError("No detections found. Cannot track.")
        tracks_df = track_cells(predictions_df, max_distance=args.max_distance)
    else:
        raise ValueError("Provide --predictions or --input-tracks")

    divisions_df = detect_divisions(
        tracks_df,
        max_gap=args.max_gap,
        max_distance_um=args.division_distance_um,
    )

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.division_edges).parent.mkdir(parents=True, exist_ok=True)
    tracks_df.to_csv(args.output, index=False)
    divisions_df.to_csv(args.division_edges, index=False)

    print(f"Tracks saved to {args.output}")
    print(f"Division edges saved to {args.division_edges}")
    print(f"Total track points: {len(tracks_df)}")
    print(f"Unique tracks: {tracks_df['track_id'].nunique()}")
    print(f"Detected division edges: {len(divisions_df)}")
    if not divisions_df.empty:
        print(f"Predicted division parent nodes: {len(divisions_df.groupby(['volume', 'source_track_id', 'source_t']))}")


if __name__ == "__main__":
    main()
