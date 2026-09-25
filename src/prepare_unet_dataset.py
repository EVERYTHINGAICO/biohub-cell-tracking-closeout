"""Prepare 3D UNet detection dataset from 199 zarr volumes + sparse .geff GT.

For each GT cell: a 32^3 patch centered on it, label = gaussian heatmap (sigma=2).
Plus --neg-ratio random negative patches per volume (no GT nearby -> zero heatmap).
Written as per-volume float16 shards (X,Y) to bound memory; 80/20 train/val split
by volume. Sparse GT is fine for DETECTION (each cell is one positive example).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import networkx as nx
import geff
from numcodecs import Blosc

CODEC = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE, blocksize=0)
P = 32          # patch size
R = P // 2
SIGMA = 2.0


def read_meta(zp: Path):
    with (zp / "0" / "zarr.json").open() as f:
        m = json.load(f)
    shape = tuple(int(v) for v in m["shape"])
    chunk = tuple(int(v) for v in m["chunk_grid"]["configuration"]["chunk_shape"])
    return shape, chunk


def read_frame(zp: Path, t: int, chunk) -> np.ndarray:
    raw = CODEC.decode((zp / "0" / "c" / str(t) / "0" / "0" / "0").read_bytes())
    return np.frombuffer(raw, dtype="<u2").reshape(chunk)[0]


def norm_frame(f: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(f, [1.0, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    return np.clip((f.astype(np.float32) - lo) / (hi - lo), 0, 1)


# precomputed gaussian kernel offsets for heatmap
_zz, _yy, _xx = np.mgrid[0:P, 0:P, 0:P]


def heatmap(cz, cy, cx):
    d2 = (_zz - cz) ** 2 + (_yy - cy) ** 2 + (_xx - cx) ** 2
    return np.exp(-d2 / (2 * SIGMA ** 2)).astype(np.float32)


def clamp_center(c, size):
    """Return patch start so a P-cube fits, and the cell offset inside the patch."""
    start = int(round(c)) - R
    start = max(0, min(start, size - P))
    return start, int(round(c)) - start


def extract_patch(frame, z, y, x):
    Z, Y, X = frame.shape
    if Z < P or Y < P or X < P:
        return None, None
    z0, cz = clamp_center(z, Z)
    y0, cy = clamp_center(y, Y)
    x0, cx = clamp_center(x, X)
    patch = frame[z0:z0 + P, y0:y0 + P, x0:x0 + P]
    return patch, (cz, cy, cx)


def process_volume(zp: Path, geff_path: Path, neg_ratio: int, rng):
    shape, chunk = read_meta(zp)
    T, Z, Y, X = shape
    g, _ = geff.read(geff_path, structure_validation=False)
    if not isinstance(g, nx.DiGraph):
        g = nx.DiGraph(g)
    by_t = {}
    for _, a in g.nodes(data=True):
        by_t.setdefault(int(a["t"]), []).append((a["z"], a["y"], a["x"]))

    Xs, Ys = [], []
    for t in sorted(by_t):
        if t >= T:
            continue
        frame = norm_frame(read_frame(zp, t, chunk))
        cells = by_t[t]
        for (z, y, x) in cells:
            patch, off = extract_patch(frame, z, y, x)
            if patch is None:
                continue
            Xs.append(patch.astype(np.float16))
            Ys.append(heatmap(*off).astype(np.float16))
        # negatives from same frame, away from cells
        cell_arr = np.array([[z, y, x] for z, y, x in cells], dtype=np.float32)
        n_neg = neg_ratio * len(cells)
        tries = 0
        got = 0
        while got < n_neg and tries < n_neg * 5:
            tries += 1
            z = rng.integers(R, Z - R); y = rng.integers(R, Y - R); x = rng.integers(R, X - R)
            if len(cell_arr) and np.min(np.abs(cell_arr - [z, y, x]).sum(1)) < P:
                continue
            patch, _ = extract_patch(frame, z, y, x)
            if patch is None:
                continue
            Xs.append(patch.astype(np.float16))
            Ys.append(np.zeros((P, P, P), np.float16))
            got += 1
    if not Xs:
        return None, None
    return np.stack(Xs), np.stack(Ys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--neg-ratio", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--val-frac", type=float, default=0.2)
    args = ap.parse_args()

    train = Path(args.train_dir)
    out = Path(args.output_dir)
    (out / "train").mkdir(parents=True, exist_ok=True)
    (out / "val").mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)

    zarrs = sorted(p for p in train.iterdir() if p.suffix == ".zarr")
    if args.limit:
        zarrs = zarrs[: args.limit]
    n_val = max(1, int(len(zarrs) * args.val_frac))
    val_set = set(z.stem for z in zarrs[:n_val])
    print(f"{len(zarrs)} volumes | {n_val} val | neg_ratio={args.neg_ratio}")

    tot_pos = tot = 0
    for i, zp in enumerate(zarrs, 1):
        name = zp.stem
        gp = train / f"{name}.geff"
        if not gp.exists():
            continue
        X, Y = process_volume(zp, gp, args.neg_ratio, rng)
        if X is None:
            print(f"  [{i}] {name}: empty"); continue
        split = "val" if name in val_set else "train"
        np.savez_compressed(out / split / f"{name}.npz", X=X, Y=Y)
        npos = int((Y.reshape(len(Y), -1).max(1) > 0.5).sum())
        tot_pos += npos; tot += len(X)
        print(f"  [{i}/{len(zarrs)}] {name} [{split}]: {len(X)} patches ({npos} pos)", flush=True)
    print(f"DONE. {tot} patches total ({tot_pos} pos) -> {out}")


if __name__ == "__main__":
    main()
