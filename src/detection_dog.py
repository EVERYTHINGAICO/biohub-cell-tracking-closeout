from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from numcodecs import Blosc
from scipy.ndimage import gaussian_filter, maximum_filter


DEFAULT_SIGMAS = (1.0, 1.5, 2.0, 2.5, 3.0)
CHUNK_CODEC = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE, blocksize=0)


def _find_zarr_paths(path: Path) -> list[Path]:
    if path.is_dir() and path.suffix == ".zarr":
        return [path]
    return sorted(p for p in path.iterdir() if p.is_dir() and p.suffix == ".zarr")


def _normalize_frame(frame: np.ndarray) -> np.ndarray:
    frame_f32 = frame.astype(np.float32, copy=False)
    lo, hi = np.percentile(frame_f32, [1.0, 99.9])
    if hi <= lo:
        return np.zeros_like(frame_f32, dtype=np.float32)
    clipped = np.clip(frame_f32, lo, hi)
    normalized = (clipped - lo) / (hi - lo)
    return normalized.astype(np.float32, copy=False)


def _read_zarr_metadata(zarr_path: Path) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    with (zarr_path / "0" / "zarr.json").open("r", encoding="utf-8") as f:
        metadata = json.load(f)
    shape = tuple(int(value) for value in metadata["shape"])
    chunk_shape = tuple(int(value) for value in metadata["chunk_grid"]["configuration"]["chunk_shape"])
    return shape, chunk_shape


def _read_timepoint(zarr_path: Path, t: int, chunk_shape: tuple[int, int, int, int]) -> np.ndarray:
    if chunk_shape[0] != 1:
        raise ValueError(f"Expected one timepoint per chunk, got {chunk_shape}")
    chunk_path = zarr_path / "0" / "c" / str(t) / "0" / "0" / "0"
    raw = CHUNK_CODEC.decode(chunk_path.read_bytes())
    return np.frombuffer(raw, dtype="<u2").reshape(chunk_shape)[0]


def _dog_response(frame: np.ndarray, sigmas: tuple[float, ...]) -> np.ndarray:
    best_response = np.zeros(frame.shape, dtype=np.float32)
    for sigma in sigmas:
        # Z voxels are physically thicker, so smooth less along Z than XY.
        sigma_small = (max(0.6, sigma * 0.35), sigma, sigma)
        sigma_large = tuple(s * 1.6 for s in sigma_small)
        small = gaussian_filter(frame, sigma=sigma_small, mode="nearest", truncate=2.5)
        large = gaussian_filter(frame, sigma=sigma_large, mode="nearest", truncate=2.5)
        response = small - large
        np.maximum(best_response, response, out=best_response)
    return best_response


def _detect_frame(
    frame: np.ndarray,
    sigmas: tuple[float, ...],
    quantile: float,
    min_score: float,
    max_detections: int,
    min_detections: int,
    xy_downsample: int,
) -> tuple[np.ndarray, np.ndarray]:
    if xy_downsample > 1:
        frame = frame[:, ::xy_downsample, ::xy_downsample]
        sigmas = tuple(max(0.5, sigma / xy_downsample) for sigma in sigmas)

    normalized = _normalize_frame(frame)
    response = _dog_response(normalized, sigmas)

    local_max = response == maximum_filter(response, size=(3, 7, 7), mode="nearest")
    threshold = max(min_score, float(np.quantile(response, quantile)))
    candidate_mask = local_max & (response >= threshold)
    coords = np.argwhere(candidate_mask)

    if len(coords) < min_detections:
        relaxed_mask = local_max & (response > 0)
        coords = np.argwhere(relaxed_mask)

    if len(coords) == 0:
        return coords, np.empty((0,), dtype=np.float32)

    scores = response[coords[:, 0], coords[:, 1], coords[:, 2]]
    order = np.argsort(scores)[::-1]
    keep = order[:max_detections]
    coords = coords[keep]
    scores = scores[keep]
    if xy_downsample > 1 and len(coords) > 0:
        coords = coords.copy()
        coords[:, 1] *= xy_downsample
        coords[:, 2] *= xy_downsample
    return coords, scores.astype(np.float32, copy=False)


def detect_volume(
    zarr_path: Path,
    sigmas: tuple[float, ...],
    quantile: float,
    min_score: float,
    max_detections_per_frame: int,
    min_detections_per_frame: int,
    xy_downsample: int,
) -> pd.DataFrame:
    shape, chunk_shape = _read_zarr_metadata(zarr_path)
    T, _, _, _ = shape
    rows = []

    for t in range(T):
        frame = _read_timepoint(zarr_path, t, chunk_shape)
        coords, scores = _detect_frame(
            frame,
            sigmas=sigmas,
            quantile=quantile,
            min_score=min_score,
            max_detections=max_detections_per_frame,
            min_detections=min_detections_per_frame,
            xy_downsample=xy_downsample,
        )
        for coord, score in zip(coords, scores):
            z, y, x = coord
            rows.append(
                {
                    "volume": zarr_path.stem,
                    "t": int(t),
                    "z": float(z),
                    "y": float(y),
                    "x": float(x),
                    "score": float(score),
                }
            )

        if (t + 1) % 20 == 0 or t == T - 1:
            print(f"  {zarr_path.name}: t={t + 1}/{T}, detections={len(rows)}", flush=True)

    return pd.DataFrame(rows, columns=["volume", "t", "z", "y", "x", "score"])


def parse_sigmas(value: str) -> tuple[float, ...]:
    return tuple(float(part.strip()) for part in value.split(",") if part.strip())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="detections_dog.csv")
    parser.add_argument("--sigmas", type=parse_sigmas, default=DEFAULT_SIGMAS)
    parser.add_argument("--quantile", type=float, default=0.9995)
    parser.add_argument("--min-score", type=float, default=0.015)
    parser.add_argument("--max-detections-per-frame", type=int, default=350)
    parser.add_argument("--min-detections-per-frame", type=int, default=40)
    parser.add_argument("--xy-downsample", type=int, default=4)
    args = parser.parse_args()

    zarr_paths = _find_zarr_paths(Path(args.test_dir))
    print(f"Found {len(zarr_paths)} zarr volumes")
    print(f"Sigmas: {args.sigmas}")

    all_detections = []
    for i, zarr_path in enumerate(zarr_paths, start=1):
        print(f"Processing {i}/{len(zarr_paths)}: {zarr_path.name}", flush=True)
        detections = detect_volume(
            zarr_path=zarr_path,
            sigmas=args.sigmas,
            quantile=args.quantile,
            min_score=args.min_score,
            max_detections_per_frame=args.max_detections_per_frame,
            min_detections_per_frame=args.min_detections_per_frame,
            xy_downsample=args.xy_downsample,
        )
        all_detections.append(detections)
        print(f"  Detections in {zarr_path.name}: {len(detections)}", flush=True)

    if all_detections:
        detections_df = pd.concat(all_detections, ignore_index=True)
    else:
        detections_df = pd.DataFrame(columns=["volume", "t", "z", "y", "x", "score"])

    detections_df.to_csv(args.output, index=False)
    print(f"Detections saved to {args.output}")
    print(f"Total detections: {len(detections_df)}")


if __name__ == "__main__":
    main()
