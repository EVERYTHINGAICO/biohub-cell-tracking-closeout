"""Train a 3D U-Net nuclei heatmap detector on the prepared patch dataset.

3-level U-Net (16-32-64), 1->1 channel, sigmoid output, MSE loss vs gaussian
heatmap targets. Adam lr=1e-3, early stopping on val loss. Saves best to
models/unet_best.pth.
"""
from __future__ import annotations
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader


class ShardDataset(Dataset):
    def __init__(self, folder: Path, max_patches: int = 200000, seed: int = 0,
                 max_shards: int = 0):
        # Keep ALL positives per shard + fill with negatives up to a per-shard
        # budget, to bound RAM (full set would be ~105GB).
        rng = np.random.default_rng(seed)
        shards = sorted(Path(folder).glob("*.npz"))
        if max_shards:
            shards = shards[:max_shards]
        per = max(1, max_patches // max(len(shards), 1))
        xs, ys = [], []
        for f in shards:
            d = np.load(f)
            X, Y = d["X"], d["Y"]
            ismax = Y.reshape(len(Y), -1).max(1) > 0.5
            pos = np.where(ismax)[0]
            neg = np.where(~ismax)[0]
            n_neg = max(0, per - len(pos))
            keep_neg = (rng.choice(neg, min(n_neg, len(neg)), replace=False)
                        if len(neg) else np.array([], int))
            idx = np.concatenate([pos, keep_neg]).astype(int)
            xs.append(X[idx]); ys.append(Y[idx])
        self.X = np.concatenate(xs).astype(np.float16)
        self.Y = np.concatenate(ys).astype(np.float16)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        x = torch.from_numpy(self.X[i].astype(np.float32))[None]
        y = torch.from_numpy(self.Y[i].astype(np.float32))[None]
        return x, y


def _block(ci, co):
    return nn.Sequential(
        nn.Conv3d(ci, co, 3, padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True),
        nn.Conv3d(co, co, 3, padding=1), nn.BatchNorm3d(co), nn.ReLU(inplace=True),
    )


class UNet3D(nn.Module):
    def __init__(self, f=(16, 32, 64)):
        super().__init__()
        self.e1 = _block(1, f[0])
        self.e2 = _block(f[0], f[1])
        self.bott = _block(f[1], f[2])
        self.pool = nn.MaxPool3d(2)
        self.up2 = nn.ConvTranspose3d(f[2], f[1], 2, stride=2)
        self.d2 = _block(f[1] * 2, f[1])
        self.up1 = nn.ConvTranspose3d(f[1], f[0], 2, stride=2)
        self.d1 = _block(f[0] * 2, f[0])
        self.out = nn.Conv3d(f[0], 1, 1)

    def forward(self, x):
        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        b = self.bott(self.pool(e2))
        d2 = self.d2(torch.cat([self.up2(b), e2], 1))
        d1 = self.d1(torch.cat([self.up1(d2), e1], 1))
        return torch.sigmoid(self.out(d1))


def run_epoch(model, loader, pos_weight, device, opt=None):
    train = opt is not None
    model.train(train)
    tot, n = 0.0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        with torch.set_grad_enabled(train):
            pred = model(x)
            # Weighted MSE: peak voxels dominate so the model can't collapse to 0
            # (heatmaps are ~99% zeros -> plain MSE just predicts all-zero).
            w = 1.0 + pos_weight * y
            loss = (w * (pred - y) ** 2).mean()
            if train:
                opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(x); n += len(x)
    return tot / max(n, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", required=True)
    ap.add_argument("--output-dir", default="models")
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--pos-weight", type=float, default=500.0)
    ap.add_argument("--max-patches", type=int, default=100000)
    ap.add_argument("--max-shards", type=int, default=0, help="0 = all shards")
    ap.add_argument("--ckpt-name", type=str, default="unet_best.pth")
    args = ap.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)

    tr = ShardDataset(Path(args.data_dir) / "train", max_patches=args.max_patches,
                      max_shards=args.max_shards)
    va = ShardDataset(Path(args.data_dir) / "val", max_patches=args.max_patches // 4,
                      max_shards=max(1, args.max_shards // 3) if args.max_shards else 0)
    print(f"train patches: {len(tr)} | val: {len(va)} | device: {device} | pos_weight={args.pos_weight}", flush=True)
    trl = DataLoader(tr, batch_size=args.batch_size, shuffle=True, num_workers=16,
                     pin_memory=True, persistent_workers=True, prefetch_factor=4)
    val = DataLoader(va, batch_size=args.batch_size, shuffle=False, num_workers=8,
                     pin_memory=True, persistent_workers=True)

    model = UNet3D().to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    best = float("inf"); bad = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl = run_epoch(model, trl, args.pos_weight, device, opt)
        vl = run_epoch(model, val, args.pos_weight, device)
        improved = vl < best - 1e-6
        if improved:
            best = vl; bad = 0
            torch.save(model.state_dict(), out / args.ckpt_name)
        else:
            bad += 1
        print(f"epoch {ep:3d}/{args.epochs} train {tl:.5f} val {vl:.5f} "
              f"best {best:.5f} {'*' if improved else ''} ({time.time()-t0:.0f}s)", flush=True)
        if bad >= args.patience:
            print(f"early stop at epoch {ep} (patience {args.patience})", flush=True)
            break
    print(f"DONE. best val {best:.5f} -> {out/'unet_best.pth'}", flush=True)


if __name__ == "__main__":
    main()
