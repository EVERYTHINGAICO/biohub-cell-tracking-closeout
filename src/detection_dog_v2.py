"""DoG cell detection v2 — physical-micrometre port of the competitor baseline.

Mirrors isakatsuyoshi/biohub-rule-based-baseline `detect_blobs`:
- physical scale in micrometres (SCALE), XY downsample 4x -> ~isotropic grid,
- multi-scale DoG over (small_um, large_um) pairs (max over scales),
- ellipsoidal max-filter footprint (radius in um),
- dual threshold: DoG response >= rel_threshold AND normalized intensity >= percentile,
- cap max_peaks per frame,
- intensity-weighted centroid refinement on the original-resolution frame.

Output CSV columns: volume, t, z, y, x, score (original voxel coords).
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numcodecs import Blosc
from scipy.ndimage import gaussian_filter, maximum_filter


# Physical voxel scale (z, y, x) in micrometres per voxel.
SCALE = np.array([1.625, 0.40625, 0.40625], dtype=np.float64)
CHUNK_CODEC = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE, blocksize=0)

# (small_um, large_um) DoG scale pairs, per competitor CONFIG_OVERRIDE.
DEFAULT_DOG_SCALES = ((1.5, 4.0), (2.2, 5.5))


def _find_zarr_paths(path: Path) -> list[Path]:
    if path.is_dir() and path.suffix == ".zarr":
        return [path]
    return sorted(p for p in path.iterdir() if p.is_dir() and p.suffix == ".zarr")


def _read_zarr_metadata(zarr_path: Path):
    with (zarr_path / "0" / "zarr.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    shape = tuple(int(v) for v in metadata["shape"])
    chunk_shape = tuple(int(v) for v in metadata["chunk_grid"]["configuration"]["chunk_shape"])
    return shape, chunk_shape


def _read_timepoint(zarr_path: Path, t: int, chunk_shape) -> np.ndarray:
    if chunk_shape[0] != 1:
        raise ValueError(f"Expected one timepoint per chunk, got {chunk_shape}")
    chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    raw = CHUNK_CODEC.decode(chunk_path.read_bytes())
    return np.frombuffer(raw, dtype="<u2").reshape(chunk_shape)[0]


def _ball_footprint(radius_um: float, eff_spacing: np.ndarray) -> np.ndarray:
    """Ellipsoidal footprint covering radius_um in physical space on the eff grid."""
    rad_vox = np.maximum(1, np.round(radius_um / eff_spacing).astype(int))
    zz, yy, xx = np.ogrid[
        -rad_vox[0]:rad_vox[0] + 1,
        -rad_vox[1]:rad_vox[1] + 1,
        -rad_vox[2]:rad_vox[2] + 1,
    ]
    d = ((zz * eff_spacing[0]) ** 2 + (yy * eff_spacing[1]) ** 2 + (xx * eff_spacing[2]) ** 2)
    return d <= radius_um ** 2


def _refine_centroids(frame: np.ndarray, coords: np.ndarray, win=(1, 3, 3)) -> np.ndarray:
    """Intensity-weighted local centre of mass refinement on original resolution."""
    if len(coords) == 0:
        return coords
    Z, Y, X = frame.shape
    ff = frame.astype(np.float64)
    out = coords.astype(np.float64).copy()
    wz, wy, wx = win
    for i, (z, y, x) in enumerate(coords):
        z, y, x = int(round(z)), int(round(y)), int(round(x))
        z0, z1 = max(0, z - wz), min(Z, z + wz + 1)
        y0, y1 = max(0, y - wy), min(Y, y + wy + 1)
        x0, x1 = max(0, x - wx), min(X, x + wx + 1)
        patch = ff[z0:z1, y0:y1, x0:x1]
        s = patch.sum()
        if s <= 0:
            continue
        zz = np.arange(z0, z1)[:, None, None]
        yy = np.arange(y0, y1)[None, :, None]
        xx = np.arange(x0, x1)[None, None, :]
        out[i, 0] = (patch * zz).sum() / s
        out[i, 1] = (patch * yy).sum() / s
        out[i, 2] = (patch * xx).sum() / s
    return out


def detect_blobs(
    frame: np.ndarray,
    xy_downsample: int,
    dog_scales,
    rel_threshold: float,
    abs_percentile: float,
    min_distance_um: float,
    max_peaks: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (coords (N,3) z,y,x in original voxels, scores (N,))."""
    vf = frame.astype(np.float32)
    ds = vf[:, ::xy_downsample, ::xy_downsample]

    # Effective physical spacing of the downsampled grid (z, y, x) -> ~isotropic.
    eff = np.array([SCALE[0], SCALE[1] * xy_downsample, SCALE[2] * xy_downsample])

    # Robust per-frame normalization (reveals dim cells).
    lo, hi = np.percentile(ds, [1.0, 99.7])
    if hi <= lo:
        hi = lo + 1.0
    norm = np.clip((ds - lo) / (hi - lo), 0, None)

    # Multi-scale DoG: max response over (small, large) um pairs.
    dog = None
    for (s_um, l_um) in dog_scales:
        resp = (gaussian_filter(norm, sigma=s_um / eff)
                - gaussian_filter(norm, sigma=l_um / eff))
        dog = resp if dog is None else np.maximum(dog, resp)

    footprint = _ball_footprint(min_distance_um, eff)
    mx = maximum_filter(dog, footprint=footprint, mode="nearest")
    abs_thr = np.percentile(norm, abs_percentile)
    peaks = (dog == mx) & (dog >= max(rel_threshold, 0.0)) & (norm >= abs_thr)
    coords = np.argwhere(peaks)
    if coords.size == 0:
        return np.zeros((0, 3), dtype=np.float64), np.zeros((0,), dtype=np.float32)

    vals = dog[peaks]
    order = np.argsort(vals)[::-1]
    coords = coords[order]
    vals = vals[order]
    if max_peaks is not None and len(coords) > max_peaks:
        coords = coords[:max_peaks]
        vals = vals[:max_peaks]

    # Map peaks back to original voxel coords.
    out = coords.astype(np.float64)
    out[:, 1] *= xy_downsample
    out[:, 2] *= xy_downsample
    return out, vals.astype(np.float32)


def detect_volume(
    zarr_path: Path,
    xy_downsample: int,
    dog_scales,
    rel_threshold: float,
    abs_percentile: float,
    min_distance_um: float,
    max_peaks: int | None,
    refine: bool,
) -> pd.DataFrame:
    shape, chunk_shape = _read_zarr_metadata(zarr_path)
    T = shape[0]
    rows = []
    for t in range(T):
        frame = _read_timepoint(zarr_path, t, chunk_shape)
        coords, scores = detect_blobs(
            frame, xy_downsample, dog_scales, rel_threshold,
            abs_percentile, min_distance_um, max_peaks,
        )
        if refine and len(coords) > 0:
            coords = _refine_centroids(frame, coords)
        for (z, y, x), score in zip(coords, scores):
            rows.append({
                "volume": zarr_path.stem, "t": int(t),
                "z": float(z), "y": float(y), "x": float(x), "score": float(score),
            })
        if (t + 1) % 20 == 0 or t == T - 1:
            print(f"  {zarr_path.name}: t={t + 1}/{T}, detections={len(rows)}", flush=True)
    return pd.DataFrame(rows, columns=["volume", "t", "z", "y", "x", "score"])


def parse_dog_scales(value: str):
    """Parse "1.5,4.0;2.2,5.5" -> ((1.5,4.0),(2.2,5.5))."""
    pairs = []
    for part in value.split(";"):
        part = part.strip()
        if not part:
            continue
        s, l = part.split(",")
        pairs.append((float(s), float(l)))
    return tuple(pairs)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="detections_dog_v2.csv")
    parser.add_argument("--xy-downsample", type=int, default=4)
    parser.add_argument("--dog-scales", type=parse_dog_scales, default=DEFAULT_DOG_SCALES)
    parser.add_argument("--rel-threshold", type=float, default=0.045)
    parser.add_argument("--abs-percentile", type=float, default=50.0)
    parser.add_argument("--min-distance-um", type=float, default=3.2)
    parser.add_argument("--max-peaks", type=int, default=40000)
    parser.add_argument("--no-refine", action="store_true",
                        help="Disable intensity-weighted centroid refinement.")
    args = parser.parse_args()

    zarr_paths = _find_zarr_paths(Path(args.test_dir))
    print(f"Found {len(zarr_paths)} zarr volumes")
    print(f"DoG scales (um): {args.dog_scales}")
    print(f"rel_threshold={args.rel_threshold} abs_percentile={args.abs_percentile} "
          f"min_distance_um={args.min_distance_um} max_peaks={args.max_peaks} "
          f"xy_downsample={args.xy_downsample} refine={not args.no_refine}")

    all_detections = []
    for i, zp in enumerate(zarr_paths, start=1):
        print(f"Processing {i}/{len(zarr_paths)}: {zp.name}", flush=True)
        det = detect_volume(
            zp, args.xy_downsample, args.dog_scales, args.rel_threshold,
            args.abs_percentile, args.min_distance_um, args.max_peaks,
            refine=not args.no_refine,
        )
        all_detections.append(det)
        print(f"  Detections in {zp.name}: {len(det)}", flush=True)

    if all_detections:
        df = pd.concat(all_detections, ignore_index=True)
    else:
        df = pd.DataFrame(columns=["volume", "t", "z", "y", "x", "score"])
    df.to_csv(args.output, index=False)
    print(f"Detections saved to {args.output}")
    print(f"Total detections: {len(df)}")


if __name__ == "__main__":
    main()
