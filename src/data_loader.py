from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import zarr
from torch.utils.data import DataLoader, Dataset


PHYSICAL_SCALE_ZYX = (1.625, 0.40625, 0.40625)
DEFAULT_VOLUME_SHAPE = (100, 64, 256, 256)
SUPPORTED_PATCH_SIZES = {
    (64, 64, 64),
    (32, 128, 128),
}


def _as_patch_size(patch_size: Sequence[int]) -> tuple[int, int, int]:
    if len(patch_size) != 3:
        raise ValueError(f"patch_size must have 3 values, got {patch_size!r}")

    patch = tuple(int(v) for v in patch_size)
    if any(v <= 0 for v in patch):
        raise ValueError(f"patch_size values must be positive, got {patch!r}")

    if patch not in SUPPORTED_PATCH_SIZES:
        raise ValueError(
            f"patch_size must be one of {sorted(SUPPORTED_PATCH_SIZES)}, got {patch!r}"
        )

    return patch


def _resolve_zarr_paths(source: str | Path | Sequence[str | Path]) -> list[Path]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if path.is_dir():
            if path.suffix == ".zarr":
                return [path]
            return sorted(p for p in path.iterdir() if p.is_dir() and p.suffix == ".zarr")
        raise FileNotFoundError(f"Path does not exist or is not a directory: {path}")

    paths = sorted(Path(p) for p in source)
    missing = [str(path) for path in paths if not path.is_dir() or path.suffix != ".zarr"]
    if missing:
        raise FileNotFoundError(f"Expected .zarr directories, got invalid paths: {missing}")
    return paths


def _axis_starts(axis_size: int, patch_size: int) -> list[int]:
    starts = list(range(0, max(axis_size - patch_size + 1, 1), patch_size))
    last_start = axis_size - patch_size
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts


def _build_patch_grid(
    volume_shape: tuple[int, int, int, int],
    patch_size: tuple[int, int, int],
) -> list[tuple[int, int, int]]:
    _, z_size, y_size, x_size = volume_shape
    pz, py, px = patch_size

    if pz > z_size or py > y_size or px > x_size:
        raise ValueError(
            f"patch_size {patch_size!r} does not fit inside spatial shape "
            f"{volume_shape[1:]!r}"
        )

    z_starts = _axis_starts(z_size, pz)
    y_starts = _axis_starts(y_size, py)
    x_starts = _axis_starts(x_size, px)

    return [(z, y, x) for z in z_starts for y in y_starts for x in x_starts]


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


def _identity_collate(batch: list[tuple[torch.Tensor, tuple[int, int, int], int]]):
    return batch[0]


class ZarrCellDataset(Dataset):
    def __init__(
        self,
        zarr_paths: str | Path | Sequence[str | Path],
        patch_size: Sequence[int] = (64, 64, 64),
        volume_shape: Sequence[int] = DEFAULT_VOLUME_SHAPE,
        time_indices: Sequence[int] | None = None,
    ) -> None:
        self.zarr_paths = _resolve_zarr_paths(zarr_paths)
        if not self.zarr_paths:
            raise ValueError("No .zarr directories found")

        self.patch_size = _as_patch_size(patch_size)
        self.volume_shape = tuple(int(v) for v in volume_shape)
        if len(self.volume_shape) != 4:
            raise ValueError(
                f"volume_shape must have 4 values (T, Z, Y, X), got {self.volume_shape!r}"
            )

        self.physical_scale_zyx = PHYSICAL_SCALE_ZYX
        self.patch_grid = _build_patch_grid(self.volume_shape, self.patch_size)
        self.total_timepoints = self.volume_shape[0]
        if time_indices is None:
            self.time_indices = list(range(self.total_timepoints))
        else:
            self.time_indices = [int(t) for t in time_indices]
            if not self.time_indices:
                raise ValueError("time_indices must not be empty")
            invalid = [t for t in self.time_indices if t < 0 or t >= self.total_timepoints]
            if invalid:
                raise ValueError(f"time_indices out of range for T={self.total_timepoints}: {invalid}")

        self.patches_per_timepoint = len(self.patch_grid)
        self.samples_per_volume = len(self.time_indices) * self.patches_per_timepoint
        self._zarr_arrays: dict[str, object] = {}

    @classmethod
    def create_temporal_splits(
        cls,
        train_frames: int = 80,
        val_frames: int = 20,
        total_timepoints: int = DEFAULT_VOLUME_SHAPE[0],
    ) -> tuple[list[int], list[int]]:
        train_frames = int(train_frames)
        val_frames = int(val_frames)
        total_timepoints = int(total_timepoints)

        if train_frames <= 0 or val_frames <= 0:
            raise ValueError("train_frames and val_frames must be positive")
        if train_frames + val_frames > total_timepoints:
            raise ValueError(
                f"train_frames + val_frames must be <= total_timepoints, got "
                f"{train_frames} + {val_frames} > {total_timepoints}"
            )

        train_indices = list(range(0, train_frames))
        val_indices = list(range(train_frames, train_frames + val_frames))
        return train_indices, val_indices

    def __len__(self) -> int:
        return len(self.zarr_paths) * self.samples_per_volume

    def _open_array(self, path: Path):
        key = str(path.resolve())
        array = self._zarr_arrays.get(key)
        if array is None:
            opened = zarr.open(str(path), mode="r")
            try:
                array = opened["0"]
            except Exception:
                array = opened
            self._zarr_arrays[key] = array
        return array

    def _decode_index(self, index: int) -> tuple[int, int, tuple[int, int, int]]:
        if index < 0 or index >= len(self):
            raise IndexError(f"Index out of range: {index}")

        volume_index, remainder = divmod(index, self.samples_per_volume)
        local_t_index, patch_index = divmod(remainder, self.patches_per_timepoint)
        t_index = self.time_indices[local_t_index]
        coords = self.patch_grid[patch_index]
        return volume_index, t_index, coords

    def __getitem__(self, index: int) -> tuple[torch.Tensor, tuple[int, int, int], int]:
        volume_index, t_index, coords = self._decode_index(index)
        z0, y0, x0 = coords
        pz, py, px = self.patch_size

        array = self._open_array(self.zarr_paths[volume_index])
        patch = np.asarray(
            array[
                t_index,
                z0 : z0 + pz,
                y0 : y0 + py,
                x0 : x0 + px,
            ]
        )
        patch = _normalize_patch(patch)
        patch_tensor = torch.from_numpy(patch).unsqueeze(0)
        assert patch_tensor.shape == (1, *self.patch_size)
        return patch_tensor, coords, t_index


def create_train_loader(
    zarr_paths: str | Path | Sequence[str | Path],
    patch_size: Sequence[int] = (64, 64, 64),
    volume_shape: Sequence[int] = DEFAULT_VOLUME_SHAPE,
    train_frames: int = 80,
    val_frames: int = 20,
) -> DataLoader:
    train_indices, _ = ZarrCellDataset.create_temporal_splits(
        train_frames=train_frames,
        val_frames=val_frames,
        total_timepoints=int(volume_shape[0]),
    )
    dataset = ZarrCellDataset(
        zarr_paths=zarr_paths,
        patch_size=patch_size,
        volume_shape=volume_shape,
        time_indices=train_indices,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=True,
        num_workers=0,
        pin_memory=True,
        collate_fn=_identity_collate,
    )


def create_val_loader(
    zarr_paths: str | Path | Sequence[str | Path],
    patch_size: Sequence[int] = (64, 64, 64),
    volume_shape: Sequence[int] = DEFAULT_VOLUME_SHAPE,
    train_frames: int = 80,
    val_frames: int = 20,
) -> DataLoader:
    _, val_indices = ZarrCellDataset.create_temporal_splits(
        train_frames=train_frames,
        val_frames=val_frames,
        total_timepoints=int(volume_shape[0]),
    )
    dataset = ZarrCellDataset(
        zarr_paths=zarr_paths,
        patch_size=patch_size,
        volume_shape=volume_shape,
        time_indices=val_indices,
    )
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
        collate_fn=_identity_collate,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("zarr_source", type=str)
    parser.add_argument(
        "--patch-size",
        type=int,
        nargs=3,
        default=(64, 64, 64),
    )
    parser.add_argument("--train-frames", type=int, default=80)
    parser.add_argument("--val-frames", type=int, default=20)
    parser.add_argument("--num-batches", type=int, default=10)
    parser.add_argument("--split", choices=("train", "val"), default="train")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    patch_size = tuple(args.patch_size)

    train_indices, val_indices = ZarrCellDataset.create_temporal_splits(
        train_frames=args.train_frames,
        val_frames=args.val_frames,
        total_timepoints=DEFAULT_VOLUME_SHAPE[0],
    )
    time_indices = train_indices if args.split == "train" else val_indices
    dataset = ZarrCellDataset(
        args.zarr_source,
        patch_size=patch_size,
        volume_shape=DEFAULT_VOLUME_SHAPE,
        time_indices=time_indices,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=args.split == "train",
        num_workers=0,
        pin_memory=True,
        collate_fn=_identity_collate,
    )

    for batch_index, (patch_tensor, coords, t_index) in enumerate(loader):
        print(
            "batch",
            batch_index,
            "shape",
            tuple(patch_tensor.shape),
            "coords",
            coords,
            "t_index",
            t_index,
        )
        assert patch_tensor.shape == (1, *patch_size)
        assert torch.isfinite(patch_tensor).all()
        if batch_index + 1 >= args.num_batches:
            break
