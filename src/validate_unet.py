"""Validate the trained 3D U-Net detector: run sliding-window inference on the
15-volume subset, extract peaks, and measure GT-node recall (<=7um) vs DoG (0.92)."""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import networkx as nx
import geff
import torch
from numcodecs import Blosc
from scipy.ndimage import maximum_filter
from scipy.spatial import cKDTree

from train_unet import UNet3D

CODEC = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE, blocksize=0)
SCALE = np.array([1.625, 0.40625, 0.40625])
P = 32
MATCH_UM = 7.0


def read_meta(zp):
    with (zp / "0" / "zarr.json").open() as f:
        m = json.load(f)
    return tuple(int(v) for v in m["shape"]), tuple(
        int(v) for v in m["chunk_grid"]["configuration"]["chunk_shape"])


def read_frame(zp, t, chunk):
    raw = CODEC.decode((zp / "0" / "c" / str(t) / "0" / "0" / "0").read_bytes())
    return np.frombuffer(raw, dtype="<u2").reshape(chunk)[0]


def norm(f):
    lo, hi = np.percentile(f, [1.0, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((f.astype(np.float32) - lo) / (hi - lo), 0, 1)


def _starts(size, stride):
    s = list(range(0, max(size - P + 1, 1), stride))
    if s[-1] != size - P:
        s.append(size - P)
    return s


@torch.no_grad()
def infer_frame(model, frame, device, stride, batch=64):
    Z, Y, X = frame.shape
    heat = np.zeros((Z, Y, X), np.float32)
    cnt = np.zeros((Z, Y, X), np.float32)
    coords, patches = [], []

    def flush():
        if not patches:
            return
        xb = torch.from_numpy(np.stack(patches).astype(np.float32))[:, None].to(device)
        pb = model(xb)[:, 0].cpu().numpy()
        for (z0, y0, x0), pr in zip(coords, pb):
            heat[z0:z0+P, y0:y0+P, x0:x0+P] += pr
            cnt[z0:z0+P, y0:y0+P, x0:x0+P] += 1
        coords.clear(); patches.clear()

    for z0 in _starts(Z, stride):
        for y0 in _starts(Y, stride):
            for x0 in _starts(X, stride):
                patches.append(frame[z0:z0+P, y0:y0+P, x0:x0+P])
                coords.append((z0, y0, x0))
                if len(patches) >= batch:
                    flush()
    flush()
    return heat / np.maximum(cnt, 1)


def peaks(heat, thr, min_dist=3):
    mx = maximum_filter(heat, size=(3, min_dist * 2 + 1, min_dist * 2 + 1))
    p = (heat == mx) & (heat >= thr)
    return np.argwhere(p)


def gt_nodes(geff_path):
    g, _ = geff.read(geff_path, structure_validation=False)
    if not isinstance(g, nx.DiGraph):
        g = nx.DiGraph(g)
    d = {}
    for _, a in g.nodes(data=True):
        d.setdefault(int(a["t"]), []).append((a["z"], a["y"], a["x"]))
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--n-vols", type=int, default=15)
    ap.add_argument("--stride", type=int, default=24)
    ap.add_argument("--thr", type=float, default=0.3)
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    model = UNet3D().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()

    train = Path(args.train_dir)
    zarrs = sorted(p for p in train.iterdir() if p.suffix == ".zarr")[: args.n_vols]

    tot = hit = ndet = 0
    for zp in zarrs:
        shape, chunk = read_meta(zp)
        T = shape[0]
        gt = gt_nodes(train / f"{zp.stem}.geff")
        for t in sorted(gt):
            if t >= T:
                continue
            heat = infer_frame(model, norm(read_frame(zp, t, chunk)), device, args.stride)
            pk = peaks(heat, args.thr)
            ndet += len(pk)
            if len(pk) == 0:
                tot += len(gt[t]); continue
            tree = cKDTree(pk * SCALE)
            for (z, y, x) in gt[t]:
                tot += 1
                dd, _ = tree.query(np.array([z, y, x]) * SCALE, k=1)
                if dd <= MATCH_UM:
                    hit += 1
        print(f"  {zp.stem}: recall so far {hit}/{tot} = {hit/max(tot,1):.3f}", flush=True)

    print(f"UNet recall: {hit/max(tot,1):.3f} ({hit}/{tot}) | detections: {ndet} | DoG=0.922")


if __name__ == "__main__":
    main()
