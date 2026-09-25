"""Base temporal linker (micrometre gate) — the competitor's link_frames, loose config.

This is the proven recall-oriented linker that reproduces the 0.826 baseline: physical
(um) Hungarian matching between consecutive frames with a generous gate, single-frame
gap bridging, and only true-isolated pruning. Kept separate from tracker_v2.py (which
owns division detection) so the two compose:

    linker.py (base tracks) -> tracker_v2.detect_divisions -> format_submission --divisions

Input CSV:  volume, t, z, y, x, score
Output CSV: volume, t, z, y, x, score, track_id
"""
from __future__ import annotations
import argparse

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment


# Physical voxel scale (z, y, x) in micrometres per voxel.
SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)


def _build_frames(volume_df: pd.DataFrame):
    if len(volume_df) == 0:
        return [], []
    T = int(volume_df["t"].max()) + 1
    frames = [np.zeros((0, 3), dtype=float) for _ in range(T)]
    fscores = [np.zeros((0,), dtype=float) for _ in range(T)]
    for t, g in volume_df.groupby("t"):
        t = int(t)
        frames[t] = g[["z", "y", "x"]].to_numpy(dtype=float)
        fscores[t] = g["score"].to_numpy(dtype=float)
    return frames, fscores


def _link_frames(frames, fscores, max_link_um):
    nodes = {}  # id -> (t, z, y, x, score)
    frame_ids = []
    nid = 1
    for t, coords in enumerate(frames):
        ids = []
        for j, (z, y, x) in enumerate(coords):
            nodes[nid] = (t, float(z), float(y), float(x), float(fscores[t][j]))
            ids.append(nid)
            nid += 1
        frame_ids.append(ids)

    edges = []
    for t in range(len(frames) - 1):
        a = frames[t]
        b = frames[t + 1]
        if len(a) == 0 or len(b) == 0:
            continue
        ap = a * SCALE
        bp = b * SCALE
        d = np.sqrt(((ap[:, None, :] - bp[None, :, :]) ** 2).sum(axis=2))
        big = max_link_um * 1000.0 + 1.0
        cost = np.where(d <= max_link_um, d, big)
        ri, ci = linear_sum_assignment(cost)
        for r, c in zip(ri, ci):
            if d[r, c] <= max_link_um:
                edges.append((frame_ids[t][r], frame_ids[t + 1][c]))
    return nodes, edges, nid


def _close_gaps(nodes, edges, next_id, max_gap, gap_dist_um):
    if not edges:
        return nodes, edges, next_id
    has_out = set(s for s, _ in edges)
    has_in = set(t for _, t in edges)

    ends_by_t = {}
    starts_by_t = {}
    for nid, (t, z, y, x, s) in nodes.items():
        if nid not in has_out:
            ends_by_t.setdefault(t, []).append(nid)
        if nid not in has_in:
            starts_by_t.setdefault(t, []).append(nid)

    new_edges = []
    for gap in range(1, max_gap + 1):
        for t, ends in ends_by_t.items():
            starts = starts_by_t.get(t + gap + 1)
            if not starts:
                continue
            ec = np.array([[nodes[e][1], nodes[e][2], nodes[e][3]] for e in ends]) * SCALE
            sc = np.array([[nodes[s][1], nodes[s][2], nodes[s][3]] for s in starts]) * SCALE
            d = np.sqrt(((ec[:, None, :] - sc[None, :, :]) ** 2).sum(axis=2))
            thr = gap_dist_um * (gap + 1)
            big = thr * 1000 + 1
            cost = np.where(d <= thr, d, big)
            ri, ci = linear_sum_assignment(cost)
            used_s = set()
            for r, c in zip(ri, ci):
                if d[r, c] > thr or ends[r] in has_out or starts[c] in used_s:
                    continue
                e_id, s_id = ends[r], starts[c]
                te, ze, ye, xe, _ = nodes[e_id]
                _, zs, ys, xs, _ = nodes[s_id]
                prev = e_id
                for k in range(1, gap + 1):
                    frac = k / (gap + 1)
                    zi = ze + (zs - ze) * frac
                    yi = ye + (ys - ye) * frac
                    xi = xe + (xs - xe) * frac
                    nid = next_id
                    next_id += 1
                    nodes[nid] = (te + k, zi, yi, xi, 0.0)
                    new_edges.append((prev, nid))
                    prev = nid
                new_edges.append((prev, s_id))
                has_out.add(e_id)
                used_s.add(s_id)
    edges = edges + new_edges
    return nodes, edges, next_id


def _assign_tracks(nodes, edges, min_track_length):
    parent = {}

    def find(a):
        parent.setdefault(a, a)
        root = a
        while parent[root] != root:
            root = parent[root]
        while parent[a] != root:
            parent[a], a = root, parent[a]
        return root

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    used = set()
    for s, t in edges:
        used.add(s)
        used.add(t)
        union(s, t)

    comp_members: dict[int, list[int]] = {}
    for nid in used:
        comp_members.setdefault(find(nid), []).append(nid)

    counter = 0
    rows = []
    for root, members in comp_members.items():
        if len(members) < min_track_length:
            continue
        counter += 1
        for nid in members:
            t, z, y, x, s = nodes[nid]
            rows.append((counter, t, z, y, x, s))
    return rows


def track_volume(volume_df, max_link_um, close_gaps, max_gap, gap_dist_um, min_track_length):
    frames, fscores = _build_frames(volume_df)
    nodes, edges, next_id = _link_frames(frames, fscores, max_link_um)
    if close_gaps:
        nodes, edges, next_id = _close_gaps(nodes, edges, next_id, max_gap, gap_dist_um)
    return _assign_tracks(nodes, edges, min_track_length)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", type=str, required=True)
    parser.add_argument("--output", type=str, default="tracks_linker.csv")
    parser.add_argument("--max-link-um", type=float, default=8.0)
    parser.add_argument("--min-track-length", type=int, default=2)
    parser.add_argument("--no-close-gaps", action="store_true",
                        help="Disable single-frame gap bridging (default ON).")
    parser.add_argument("--max-gap", type=int, default=1)
    parser.add_argument("--gap-dist-um", type=float, default=6.0)
    args = parser.parse_args()

    predictions_df = pd.read_csv(args.predictions)
    print(f"Loaded {len(predictions_df)} detections")
    if len(predictions_df) == 0:
        print("ERROR: No detections found. Cannot track.")
        return

    close_gaps = not args.no_close_gaps
    print(f"Config: max_link_um={args.max_link_um} min_track_length={args.min_track_length} "
          f"close_gaps={close_gaps} max_gap={args.max_gap} gap_dist_um={args.gap_dist_um}")

    out_rows = []
    for volume, volume_df in predictions_df.groupby("volume", sort=True):
        rows = track_volume(
            volume_df,
            max_link_um=args.max_link_um,
            close_gaps=close_gaps,
            max_gap=args.max_gap,
            gap_dist_um=args.gap_dist_um,
            min_track_length=args.min_track_length,
        )
        for track_id, t, z, y, x, s in rows:
            out_rows.append({
                "volume": volume, "t": int(t), "z": z, "y": y, "x": x,
                "score": s, "track_id": int(track_id),
            })
        n_tracks = len({r[0] for r in rows})
        print(f"  {volume}: nodes={len(rows)} tracks={n_tracks}", flush=True)

    tracks_df = pd.DataFrame(
        out_rows, columns=["volume", "t", "z", "y", "x", "score", "track_id"]
    )
    tracks_df.to_csv(args.output, index=False)
    print(f"Tracks saved to {args.output}")
    print(f"Total track points: {len(tracks_df)}")
    if len(tracks_df):
        lengths = tracks_df.groupby(["volume", "track_id"]).size()
        n_unique = tracks_df.groupby("volume")["track_id"].nunique().sum()
        print(f"Unique tracks (summed per volume): {n_unique}")
        print(f"Track length: median={lengths.median():.1f} mean={lengths.mean():.1f} "
              f"max={lengths.max()} min={lengths.min()}")


if __name__ == "__main__":
    main()
