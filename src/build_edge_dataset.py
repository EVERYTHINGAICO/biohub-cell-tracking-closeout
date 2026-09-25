"""Build a learned-linker training set (OUR OWN, not copied): candidate detection
pairs between consecutive frames, labelled by the GT tracking graph.

For each volume: match DoG detections to GT nodes (<=7um). For consecutive frames
t,t+1, form candidate pairs within a physical gate; a pair is POSITIVE iff both
detections match GT nodes joined by a GT edge, else NEGATIVE. Geometric features
now; appearance (image patches) is the planned next addition.

Output: /tmp/edge_dataset/<vol>.npz with X (features) and y (labels).
Features: [d_um, dz_um, dy_um, dx_um, score_i, score_j].
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd, geff, networkx as nx
from scipy.spatial import cKDTree

SCALE = np.array([1.625, 0.40625, 0.40625])
MATCH_UM = 7.0
GATE_UM = 12.0


def gt_data(geff_path):
    g, _ = geff.read(geff_path, structure_validation=False)
    if not isinstance(g, nx.DiGraph):
        g = nx.DiGraph(g)
    nodes = {}
    for n, a in g.nodes(data=True):
        nodes.setdefault(int(a["t"]), []).append((int(n), a["z"], a["y"], a["x"]))
    edges = {(int(u), int(v)) for u, v in g.edges()}
    return nodes, edges


def match_frame(dets_zyx, gt_list):
    """Return array mapping each det -> gt node id (or -1) within MATCH_UM."""
    if not gt_list or len(dets_zyx) == 0:
        return np.full(len(dets_zyx), -1, dtype=np.int64)
    gt_ids = np.array([g[0] for g in gt_list])
    gt_xyz = np.array([[g[1], g[2], g[3]] for g in gt_list]) * SCALE
    tree = cKDTree(gt_xyz)
    d, idx = tree.query(dets_zyx * SCALE, k=1)
    out = np.where(d <= MATCH_UM, gt_ids[idx], -1)
    return out


def build_volume(dets_v, gt_nodes, gt_edges):
    frames = {}
    for t, g in dets_v.groupby("t"):
        frames[int(t)] = (g[["z", "y", "x"]].to_numpy(float), g["score"].to_numpy(float))
    X, y = [], []
    for t in sorted(frames):
        if (t + 1) not in frames:
            continue
        zi, si = frames[t]; zj, sj = frames[t + 1]
        if len(zi) == 0 or len(zj) == 0:
            continue
        mi = match_frame(zi, gt_nodes.get(t, []))
        mj = match_frame(zj, gt_nodes.get(t + 1, []))
        pi = zi * SCALE; pj = zj * SCALE
        treej = cKDTree(pj)
        for i in range(len(zi)):
            for j in treej.query_ball_point(pi[i], GATE_UM):
                dv = pi[i] - pj[j]
                dist = float(np.linalg.norm(dv))
                lbl = 1 if (mi[i] != -1 and mj[j] != -1 and (int(mi[i]), int(mj[j])) in gt_edges) else 0
                X.append([dist, dv[0], dv[1], dv[2], si[i], sj[j]])
                y.append(lbl)
    return np.array(X, np.float32), np.array(y, np.int8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--dets", default="detections_train.csv")
    ap.add_argument("--output-dir", default="/tmp/edge_dataset")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    train = Path(args.train_dir)
    print("loading detections...", flush=True)
    dets = pd.read_csv(args.dets)
    vols = sorted(dets["volume"].unique())
    if args.limit:
        vols = vols[: args.limit]

    tot_pos = tot = 0
    for i, v in enumerate(vols, 1):
        gp = train / f"{v}.geff"
        if not gp.exists():
            continue
        gn, ge = gt_data(gp)
        X, yy = build_volume(dets[dets.volume == v], gn, ge)
        np.savez_compressed(out / f"{v}.npz", X=X, y=yy)
        npos = int(yy.sum())
        tot_pos += npos; tot += len(yy)
        print(f"[{i}/{len(vols)}] {v}: {len(yy)} pairs ({npos} pos, {npos/max(len(yy),1):.3%})", flush=True)
    print(f"DONE. {tot} pairs total, {tot_pos} positive ({tot_pos/max(tot,1):.3%}) -> {out}")


if __name__ == "__main__":
    main()
