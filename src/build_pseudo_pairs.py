"""STEP 2a: build patch-pair dataset from DENSE pseudo-labels (pilkwang tracks).

Positives: consecutive-frame node pairs within the same pseudo track_id.
Hard negatives: for each positive source, other t+1 detections within GATE_UM but
in a different track (ratio 1:NEG). Each endpoint's 32^3 image patch stored once
(float16); pairs reference patches by index. Per-volume npz saved to /tmp/pairs.
"""
from __future__ import annotations
import argparse, json
from pathlib import Path
import numpy as np, pandas as pd
from numcodecs import Blosc
from scipy.spatial import cKDTree

CODEC = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE, blocksize=0)
SCALE = np.array([1.625, 0.40625, 0.40625]); GATE_UM = 12.0; NEG = 5
P = 32; R = P // 2

def read_meta(zp):
    m = json.load(open(zp/"0"/"zarr.json"))
    return tuple(int(v) for v in m["shape"]), tuple(int(v) for v in m["chunk_grid"]["configuration"]["chunk_shape"])
def read_frame(zp, t, ch):
    raw = CODEC.decode((zp/"0"/"c"/str(t)/"0"/"0"/"0").read_bytes())
    return np.frombuffer(raw, dtype="<u2").reshape(ch)[0]
def norm(f):
    lo, hi = np.percentile(f, [1.0, 99.7]); hi = hi if hi > lo else lo + 1
    return np.clip((f.astype(np.float32)-lo)/(hi-lo), 0, 1).astype(np.float32)
def patch(fr, z, y, x):
    Z, Y, X = fr.shape
    z0 = min(max(int(round(z))-R, 0), Z-P); y0 = min(max(int(round(y))-R, 0), Y-P); x0 = min(max(int(round(x))-R, 0), X-P)
    return fr[z0:z0+P, y0:y0+P, x0:x0+P]

def build_volume(vol, df, train_dir, out_dir, rng):
    zp = Path(train_dir)/f"{vol}.zarr"
    if not zp.exists(): return None
    _, ch = read_meta(zp)
    sub = df[df.volume_id == vol]
    by_t = {t: g for t, g in sub.groupby("t")}
    ts = sorted(by_t)
    # index nodes: node_id -> (t,z,y,x,track)
    node = {int(r.node_id): (int(r.t), r.z, r.y, r.x, int(r.track_id)) for r in sub.itertuples()}
    patches, pidx = [], {}   # node_id -> patch index
    pairs = []               # [i, j, label]
    frame_cache = {}
    def get_patch(nid):
        if nid in pidx: return pidx[nid]
        t, z, y, x, _ = node[nid]
        if t not in frame_cache: frame_cache[t] = norm(read_frame(zp, t, ch))
        patches.append(patch(frame_cache[t], z, y, x).astype(np.float16))
        pidx[nid] = len(patches)-1; return pidx[nid]
    for t in ts:
        if t+1 not in by_t: continue
        cur = by_t[t]; nxt = by_t[t+1]
        nxt_ids = nxt.node_id.values.astype(int)
        nxt_zyx = nxt[["z","y","x"]].values * SCALE
        nxt_track = nxt.track_id.values.astype(int)
        tree = cKDTree(nxt_zyx)
        for r in cur.itertuples():
            src = int(r.node_id); src_tr = int(r.track_id)
            src_zyx = np.array([r.z, r.y, r.x]) * SCALE
            # positive: same track successor in t+1
            pos_mask = nxt_track == src_tr
            if not pos_mask.any(): continue
            pos_id = int(nxt_ids[pos_mask][0])
            i = get_patch(src); j = get_patch(pos_id); pairs.append((i, j, 1))
            # hard negatives: within gate, different track
            cand = tree.query_ball_point(src_zyx, GATE_UM)
            negs = [k for k in cand if nxt_track[k] != src_tr]
            if negs:
                rng.shuffle(negs)
                for k in negs[:NEG]:
                    jn = get_patch(int(nxt_ids[k])); pairs.append((i, jn, 0))
    if not pairs: return None
    patches = np.stack(patches).astype(np.float16)
    pairs = np.array(pairs, dtype=np.int32)
    np.savez_compressed(Path(out_dir)/f"{vol}.npz", patches=patches, pairs=pairs)
    return len(patches), int((pairs[:,2]==1).sum()), int((pairs[:,2]==0).sum())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="/tmp/pseudolabels/pseudo_gt_batch20.csv")
    ap.add_argument("--train-dir", default="/mnt/d/descargas al dico vergas/biohub-cell-tracking-during-development/train")
    ap.add_argument("--out", default="/tmp/pairs")
    args = ap.parse_args()
    Path(args.out).mkdir(exist_ok=True)
    df = pd.read_csv(args.csv)
    rng = np.random.default_rng(0)
    tot_p = tot_pos = tot_neg = 0
    for vol in sorted(df.volume_id.unique()):
        res = build_volume(vol, df, args.train_dir, args.out, rng)
        if res:
            np_, pos, neg = res; tot_p += np_; tot_pos += pos; tot_neg += neg
            print(f"  {vol}: {np_} patches, {pos} pos, {neg} neg", flush=True)
    print(f"TOTAL: {tot_p} patches, {tot_pos} pos, {tot_neg} neg -> {args.out}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
