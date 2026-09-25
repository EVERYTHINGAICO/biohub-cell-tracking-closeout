from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np
import torch
import zarr
import pandas as pd
from unet3d import UNet3D


def _normalize_patch(patch: np.ndarray) -> np.ndarray:
    patch_f32 = patch.astype(np.float32, copy=False)
    lo, hi = np.percentile(patch_f32, [1.0, 99.9])

    if hi <= lo:
        clipped = np.clip(patch_f32, lo, lo)
    else:
        clipped = np.clip(patch_f32, lo, hi)

    mean = float(clipped.mean())
    std = float(clipped.std())
    if std < 1e-6:
        std = 1.0

    normalized = (clipped - mean) / std
    normalized = normalized.astype(np.float32, copy=False)
    assert np.isfinite(normalized).all()
    return normalized


def _find_zarr_paths(path: Path) -> list[Path]:
    if path.is_dir() and path.suffix == ".zarr":
        return [path]
    return sorted(p for p in path.iterdir() if p.is_dir() and p.suffix == ".zarr")


def detect_cells_in_volume(
    model,
    zarr_path,
    device,
    threshold=0.1,
    patch_size=(64, 64, 64),
    max_timepoints=None,
    batch_size=8,
):
    zarr_store = zarr.open(str(zarr_path), mode="r")
    volume = zarr_store["0"]
    T, D, H, W = volume.shape
    if max_timepoints is not None:
        T = min(T, int(max_timepoints))

    all_detections = []
    stride = tuple(max(1, s // 2) for s in patch_size)

    def flush_batch(batch_patches, batch_infos):
        if not batch_patches:
            return

        batch_np = np.stack(batch_patches, axis=0)
        patch_tensor = torch.from_numpy(batch_np).float().unsqueeze(1).to(device)

        with torch.no_grad():
            if device.type == "cuda":
                with torch.amp.autocast(device_type="cuda"):
                    _, probabilities = model(patch_tensor)
            else:
                _, probabilities = model(patch_tensor)
            prob_batch = probabilities.detach().cpu().numpy()[:, 0]

        for prob_np, info in zip(prob_batch, batch_infos):
            t, z, y, x, z_end, y_end, x_end = info
            coords = np.argwhere(prob_np > threshold)
            if len(coords) > 0:
                scores = prob_np[coords[:, 0], coords[:, 1], coords[:, 2]]
                for coord, score in zip(coords, scores):
                    cz, cy, cx = coord
                    if z + cz < z_end and y + cy < y_end and x + cx < x_end:
                        all_detections.append(
                            {
                                "t": t,
                                "z": float(z + cz),
                                "y": float(y + cy),
                                "x": float(x + cx),
                                "score": float(score),
                            }
                        )

        batch_patches.clear()
        batch_infos.clear()

    for t in range(T):
        frame = volume[t]
        batch_patches = []
        batch_infos = []

        for z in range(0, D, stride[0]):
            for y in range(0, H, stride[1]):
                for x in range(0, W, stride[2]):
                    z_end = min(z + patch_size[0], D)
                    y_end = min(y + patch_size[1], H)
                    x_end = min(x + patch_size[2], W)

                    patch = np.asarray(frame[z:z_end, y:y_end, x:x_end])

                    if patch.shape != patch_size:
                        padded = np.zeros(patch_size, dtype=patch.dtype)
                        padded[: patch.shape[0], : patch.shape[1], : patch.shape[2]] = patch
                        patch = padded

                    patch_normalized = _normalize_patch(patch)
                    batch_patches.append(patch_normalized)
                    batch_infos.append((t, z, y, x, z_end, y_end, x_end))

                    if len(batch_patches) >= batch_size:
                        flush_batch(batch_patches, batch_infos)

        flush_batch(batch_patches, batch_infos)

    if len(all_detections) > 0:
        df = pd.DataFrame(all_detections)
        df["z_int"] = df["z"].round().astype(int)
        df["y_int"] = df["y"].round().astype(int)
        df["x_int"] = df["x"].round().astype(int)
        df = (
            df.sort_values("score", ascending=False)
            .drop_duplicates(subset=["t", "z_int", "y_int", "x_int"])
            .drop(columns=["z_int", "y_int", "x_int"])
        )
        return df

    return pd.DataFrame(columns=["t", "z", "y", "x", "score"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--test-dir", type=str, required=True)
    parser.add_argument("--output", type=str, default="predictions.csv")
    parser.add_argument("--threshold", type=float, default=0.3)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-volumes", type=int, default=None)
    parser.add_argument("--max-timepoints", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=8)
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    model = UNet3D().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))
    model.eval()
    print(f"Model loaded from {args.model}")

    zarr_paths = _find_zarr_paths(Path(args.test_dir))
    if args.max_volumes is not None:
        zarr_paths = zarr_paths[: args.max_volumes]
    print(f"Found {len(zarr_paths)} test volumes")

    all_predictions = []
    for i, zarr_path in enumerate(zarr_paths):
        print(f"Processing {i + 1}/{len(zarr_paths)}: {zarr_path.name}")
        detections = detect_cells_in_volume(
            model,
            zarr_path,
            device,
            args.threshold,
            max_timepoints=args.max_timepoints,
            batch_size=args.batch_size,
        )
        detections["volume"] = zarr_path.stem
        all_predictions.append(detections)
        print(f"  Detections: {len(detections)}")

    if len(all_predictions) > 0:
        predictions_df = pd.concat(all_predictions, ignore_index=True)
    else:
        predictions_df = pd.DataFrame(columns=["t", "z", "y", "x", "score", "volume"])

    predictions_df.to_csv(args.output, index=False)
    print(f"Predictions saved to {args.output}")
    print(f"Total detections: {len(predictions_df)}")


if __name__ == "__main__":
    main()
