"""FASE 2: train a 3D siamese-style appearance encoder to score whether two
detection patches are the SAME cell across consecutive frames.

Input: two 32^3 patches (stacked as 2 channels) + geometric features.
Output: edge probability (same cell -> 1). Loss: BCE with pos_weight for the
1:5 imbalance. Target AUC > 0.90 (beat the geometric-only GBM at 0.85).
"""
from __future__ import annotations
import argparse, glob, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score


class PairDataset(Dataset):
    """Loads appearance shards; each item = (2ch patch, geom feats, label)."""
    def __init__(self, files):
        self.P, self.pairs = [], []
        offset = 0
        for f in files:
            d = np.load(f)
            self.P.append(d["patches"])
            pr = d["pairs"].copy()
            pr[:, 0] += offset; pr[:, 1] += offset  # global patch indices
            offset += len(d["patches"])
            self.pairs.append(pr)
        self.P = np.concatenate(self.P).astype(np.float16)
        self.pairs = np.concatenate(self.pairs).astype(np.float32)
        # normalize geom features (cols 3..8) by train stats
        self.gmean = self.pairs[:, 3:].mean(0); self.gstd = self.pairs[:, 3:].std(0) + 1e-6

    def __len__(self): return len(self.pairs)

    def __getitem__(self, k):
        r = self.pairs[k]
        i, j, lbl = int(r[0]), int(r[1]), r[2]
        x = np.stack([self.P[i], self.P[j]]).astype(np.float32)  # (2,32,32,32)
        g = (r[3:] - self.gmean) / self.gstd
        return torch.from_numpy(x), torch.from_numpy(g.astype(np.float32)), torch.tensor(lbl, dtype=torch.float32)


def _blk(ci, co):
    return nn.Sequential(nn.Conv3d(ci, co, 3, padding=1), nn.BatchNorm3d(co), nn.ReLU(True),
                         nn.Conv3d(co, co, 3, padding=1), nn.BatchNorm3d(co), nn.ReLU(True), nn.MaxPool3d(2))


class AppearanceNet(nn.Module):
    def __init__(self, n_geom=6):
        super().__init__()
        self.enc = nn.Sequential(_blk(2, 16), _blk(16, 32), _blk(32, 64))  # 32->4
        self.head = nn.Sequential(nn.Linear(64 * 4 * 4 * 4 + n_geom, 128), nn.ReLU(True),
                                  nn.Dropout(0.3), nn.Linear(128, 1))

    def forward(self, x, g):
        h = self.enc(x).flatten(1)
        return self.head(torch.cat([h, g], 1)).squeeze(1)


def run(model, loader, dev, crit, opt=None):
    model.train(opt is not None); tot = 0; n = 0; P = []; Y = []
    for x, g, y in loader:
        x, g, y = x.to(dev), g.to(dev), y.to(dev)
        with torch.set_grad_enabled(opt is not None):
            logit = model(x, g); loss = crit(logit, y)
            if opt: opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item() * len(y); n += len(y)
        P.append(torch.sigmoid(logit).detach().cpu().numpy()); Y.append(y.cpu().numpy())
    P = np.concatenate(P); Y = np.concatenate(Y)
    auc = roc_auc_score(Y, P) if Y.min() != Y.max() else 0.5
    return tot / n, auc, average_precision_score(Y, P)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", default="/tmp/appearance_dataset")
    ap.add_argument("--output", default="models/appearance_best.pth")
    ap.add_argument("--epochs", type=int, default=20); ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3); ap.add_argument("--patience", type=int, default=5)
    ap.add_argument("--pos-weight", type=float, default=5.0)
    args = ap.parse_args()

    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(f"{args.data_dir}/*.npz"))
    nval = max(1, len(files) // 5); vfiles, tfiles = files[:nval], files[nval:]
    tr, va = PairDataset(tfiles), PairDataset(vfiles)
    va.gmean, va.gstd = tr.gmean, tr.gstd
    print(f"train pairs {len(tr)} | val pairs {len(va)} | device {dev}", flush=True)
    trl = DataLoader(tr, args.batch_size, shuffle=True, num_workers=8, pin_memory=True)
    val = DataLoader(va, args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    model = AppearanceNet().to(dev); opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    crit = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(args.pos_weight, device=dev))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)

    best = 0.0; bad = 0
    for ep in range(1, args.epochs + 1):
        t0 = time.time()
        tl, ta, tap = run(model, trl, dev, crit, opt)
        vl, vauc, vap = run(model, val, dev, crit)
        imp = vauc > best + 1e-4
        if imp: best = vauc; bad = 0; torch.save(model.state_dict(), args.output)
        else: bad += 1
        print(f"ep {ep:2d} train_auc {ta:.4f} VAL_AUC {vauc:.4f} val_ap {vap:.4f} best {best:.4f} "
              f"{'*' if imp else ''} ({time.time()-t0:.0f}s)", flush=True)
        if bad >= args.patience: print(f"early stop ep {ep}"); break
    print(f"DONE. best VAL AUC {best:.4f} (geom GBM baseline 0.849) -> {args.output}")


if __name__ == "__main__":
    main()
