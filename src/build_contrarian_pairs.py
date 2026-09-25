"""CONTRARIAN dataset: labels from REAL sparse GT (not pilkwang) where GT exists.

For each vol: match real GT nodes -> pilkwang pseudo nodes (7um). Then:
 - POSITIVE = real GT edge (gt_src->gt_dst mapped to pseudo nodes). These include
   the ~417 edges pilkwang MISSES (its blind spots) -> the diversity signal.
 - NEGATIVE = pilkwang's own successor of a GT-matched source when it CONTRADICTS
   the real GT successor (pilkwang wrong per GT), plus nearby diff-cell GT non-edges.
All pairs carry high weight so a fine-tune forces the student to DISAGREE with
pilkwang exactly where the real GT does. Saved to /tmp/contrarian/<vol>.npz.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd, geff, networkx as nx
from scipy.spatial import cKDTree
from build_pseudo_pairs import read_meta, read_frame, norm, patch, SCALE

MATCH_UM = 7.0; GATE_UM = 12.0; P = 32

def gt_nodes_edges(gp):
    g, _ = geff.read(gp, structure_validation=False)
    if not isinstance(g, nx.DiGraph): g = nx.DiGraph(g)
    nodes = {int(n): (int(a["t"]), float(a["z"]), float(a["y"]), float(a["x"])) for n, a in g.nodes(data=True)}
    edges = {(int(u), int(v)) for u, v in g.edges()}
    return nodes, edges

def build_volume(vol, df, train_dir, gt_dir, rng):
    zp = Path(train_dir)/f"{vol}.zarr"; gp = Path(gt_dir)/f"{vol}.geff"
    if not zp.exists() or not gp.exists(): return None
    _, ch = read_meta(zp)
    sub = df[df.volume_id == vol]
    # pseudo nodes indexed by t
    ps_by_t = {}   # t -> (ids array, zyx*scale array, track array)
    pnode = {}     # pseudo node_id -> (t,z,y,x,track)
    for t, gr in sub.groupby("t"):
        ids = gr.node_id.values.astype(int)
        ps_by_t[int(t)] = (ids, gr[["z","y","x"]].values*SCALE, gr.track_id.values.astype(int))
        for r in gr.itertuples(): pnode[int(r.node_id)] = (int(r.t), r.z, r.y, r.x, int(r.track_id))
    gnodes, gedges = gt_nodes_edges(gp)
    # match GT node -> pseudo node (same t, <=7um)
    g2p = {}
    for gid, (t, z, y, x) in gnodes.items():
        if t not in ps_by_t: continue
        ids, zyx, _ = ps_by_t[t]
        d = np.linalg.norm(zyx - np.array([z, y, x])*SCALE, axis=1)
        k = int(np.argmin(d))
        if d[k] <= MATCH_UM: g2p[gid] = int(ids[k])
    # pilkwang successor of a pseudo node = same track, t+1
    def pilk_succ(pid):
        t, z, y, x, tr = pnode[pid]
        if t+1 not in ps_by_t: return None
        ids, _, trk = ps_by_t[t+1]
        m = np.where(trk == tr)[0]
        return int(ids[m[0]]) if len(m) else None
    pairs = []  # (src_pid, dst_pid, label)
    for (gs, gd) in gedges:
        if gs in g2p and gd in g2p:
            ps, pd_ = g2p[gs], g2p[gd]
            pairs.append((ps, pd_, 1))              # real-GT positive (incl pilkwang FN)
            psucc = pilk_succ(ps)
            if psucc is not None and psucc != pd_:
                pairs.append((ps, psucc, 0))        # pilkwang's contradicting edge = negative
    if not pairs: return None
    # extract patches for all involved pseudo nodes
    need = sorted({p for a, b, _ in pairs for p in (a, b)})
    frame_cache, pidx, patches = {}, {}, []
    for pid in need:
        t, z, y, x, _ = pnode[pid]
        if t not in frame_cache: frame_cache[t] = norm(read_frame(zp, t, ch))
        patches.append(patch(frame_cache[t], z, y, x).astype(np.float16)); pidx[pid] = len(patches)-1
    P_ = np.stack(patches).astype(np.float16)
    pr = np.array([[pidx[a], pidx[b], l] for a, b, l in pairs], dtype=np.int32)
    np.savez_compressed(Path("/tmp/contrarian")/f"{vol}.npz", patches=P_, pairs=pr)
    return len(P_), int((pr[:,2]==1).sum()), int((pr[:,2]==0).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/tmp/pseudolabels/pseudo_gt_batch20.csv")
    ap.add_argument("--train-dir", default="/mnt/d/descargas al dico vergas/biohub-cell-tracking-during-development/train")
    args = ap.parse_args()
    Path("/tmp/contrarian").mkdir(exist_ok=True)
    df = pd.read_csv(args.csv); rng = np.random.default_rng(0)
    tp = tpos = tneg = 0
    for vol in sorted(df.volume_id.unique()):
        r = build_volume(vol, df, args.train_dir, args.train_dir, rng)
        if r: n, pos, neg = r; tp += n; tpos += pos; tneg += neg; print(f"  {vol}: {pos} pos, {neg} neg", flush=True)
    print(f"TOTAL contrarian: {tpos} pos, {tneg} neg, {tp} patches", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
