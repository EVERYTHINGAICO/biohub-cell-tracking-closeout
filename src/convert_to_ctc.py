"""Convert Biohub zarr volumes + .geff GT into CTC format for trackastra training.

Per volume <seq> writes:
  <out>/<seq>/img/t{t:03d}.tif         raw 3D image frame (Z,Y,X) uint16
  <out>/<seq>/TRA/man_track{t:03d}.tif  3D int label mask (each GT track = its label)
  <out>/<seq>/TRA/man_track.txt         rows "L B E P" (label, begin, end, parent)

GT is sparse (~1-2 annotated lineages/frame): each GT node is painted as a small
ellipsoid labelled with its CTC track id. Tracks are the maximal non-branching
lineage segments of the .geff graph; divisions set the child's parent label.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import networkx as nx
import geff
import tifffile
from numcodecs import Blosc

CODEC = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE, blocksize=0)


def read_meta(zp: Path):
    with (zp / "0" / "zarr.json").open() as f:
        m = json.load(f)
    shape = tuple(int(v) for v in m["shape"])
    chunk = tuple(int(v) for v in m["chunk_grid"]["configuration"]["chunk_shape"])
    return shape, chunk


def read_frame(zp: Path, t: int, chunk) -> np.ndarray:
    raw = CODEC.decode((zp / "0" / "c" / str(t) / "0" / "0" / "0").read_bytes())
    return np.frombuffer(raw, dtype="<u2").reshape(chunk)[0]


def load_gt_graph(geff_path: Path) -> nx.DiGraph:
    g, _ = geff.read(geff_path, structure_validation=False)
    if not isinstance(g, nx.DiGraph):
        g = nx.DiGraph(g)
    return g


def decompose_tracklets(g: nx.DiGraph):
    """Return node2track dict and list of (label, begin, end, parent)."""
    node_t = {n: int(a["t"]) for n, a in g.nodes(data=True)}
    # a node starts a tracklet if it has no parent, or its parent divides (>1 child)
    starts = []
    for n in g.nodes:
        preds = list(g.predecessors(n))
        if not preds or g.out_degree(preds[0]) >= 2:
            starts.append(n)
    starts.sort(key=lambda n: node_t[n])  # temporal order so parents assigned first

    node2track = {}
    rows = []
    label = 0
    for s in starts:
        label += 1
        preds = list(g.predecessors(s))
        parent = node2track[preds[0]] if (preds and g.out_degree(preds[0]) >= 2) else 0
        cur = s
        frames = []
        while True:
            node2track[cur] = label
            frames.append(node_t[cur])
            succ = list(g.successors(cur))
            if len(succ) == 1 and g.out_degree(cur) == 1 and g.in_degree(succ[0]) == 1:
                cur = succ[0]
            else:
                break
        rows.append((label, min(frames), max(frames), parent))
    return node2track, rows


def paint(mask, z, y, x, label, rz, ry, rx):
    Z, Y, X = mask.shape
    z, y, x = int(round(z)), int(round(y)), int(round(x))
    zz = slice(max(0, z - rz), min(Z, z + rz + 1))
    yy = slice(max(0, y - ry), min(Y, y + ry + 1))
    xx = slice(max(0, x - rx), min(X, x + rx + 1))
    mask[zz, yy, xx] = label


def convert_volume(zp: Path, geff_path: Path, out: Path, radii):
    shape, chunk = read_meta(zp)
    T, Z, Y, X = shape
    g = load_gt_graph(geff_path)
    node2track, rows = decompose_tracklets(g)

    img_dir = out / "img"
    tra_dir = out / "TRA"
    img_dir.mkdir(parents=True, exist_ok=True)
    tra_dir.mkdir(parents=True, exist_ok=True)

    # group GT nodes by frame
    by_t = {}
    for n, a in g.nodes(data=True):
        by_t.setdefault(int(a["t"]), []).append((n, a["z"], a["y"], a["x"]))

    rz, ry, rx = radii
    for t in range(T):
        frame = read_frame(zp, t, chunk)
        tifffile.imwrite(img_dir / f"t{t:03d}.tif", frame.astype(np.uint16))
        mask = np.zeros((Z, Y, X), dtype=np.int32)
        for n, z, y, x in by_t.get(t, []):
            paint(mask, z, y, x, node2track[n], rz, ry, rx)
        tifffile.imwrite(tra_dir / f"man_track{t:03d}.tif", mask)

    with (tra_dir / "man_track.txt").open("w") as f:
        for label, b, e, p in rows:
            f.write(f"{label} {b} {e} {p}\n")
    return T, len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--limit", type=int, default=0, help="max volumes (0 = all)")
    ap.add_argument("--radii", type=int, nargs=3, default=(2, 4, 4), help="z y x ball radii")
    args = ap.parse_args()

    train = Path(args.train_dir)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    zarrs = sorted(p for p in train.iterdir() if p.suffix == ".zarr")
    if args.limit:
        zarrs = zarrs[: args.limit]
    print(f"converting {len(zarrs)} volumes -> {out}")

    total_tracks = 0
    for i, zp in enumerate(zarrs, 1):
        name = zp.stem
        geff_path = train / f"{name}.geff"
        if not geff_path.exists():
            print(f"  [{i}] SKIP {name}: no geff")
            continue
        T, ntr = convert_volume(zp, geff_path, out / name, tuple(args.radii))
        total_tracks += ntr
        print(f"  [{i}/{len(zarrs)}] {name}: {T} frames, {ntr} tracklets", flush=True)
    print(f"DONE. {len(zarrs)} volumes, {total_tracks} tracklets total -> {out}")


if __name__ == "__main__":
    main()
