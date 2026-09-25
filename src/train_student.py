"""STEP 2b: train a ResNet3D siamese association model on DENSE pseudo-labels.

Distillation of pilkwang's linker: input = two 32^3 patches (2ch), output =
P(same cell). ResNet3D encoder + focal loss + 3D augmentation. Trains on N-4
volumes, validates on 4 held-out. Val AUC measures how well the student
reproduces the teacher (ceiling = pilkwang). Prior sparse-GT encoder overfit at
AUC 0.828; the question is whether 632k dense labels push it materially higher.
"""
from __future__ import annotations
import argparse, glob, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import roc_auc_score, average_precision_score

class PairDS(Dataset):
    def __init__(self, files, augment=False, max_pairs=0, seed=0):
        self.P, self.pairs = [], []; off = 0
        for f in files:
            d = np.load(f); self.P.append(d["patches"])
            pr = d["pairs"].astype(np.int64).copy(); pr[:, 0] += off; pr[:, 1] += off
            off += len(d["patches"]); self.pairs.append(pr)
        self.P = np.concatenate(self.P); self.pairs = np.concatenate(self.pairs)
        if max_pairs and len(self.pairs) > max_pairs:  # subsample, keep class ratio
            rng = np.random.default_rng(seed)
            pos = np.where(self.pairs[:, 2] == 1)[0]; neg = np.where(self.pairs[:, 2] == 0)[0]
            frac = max_pairs / len(self.pairs)
            keep = np.concatenate([rng.choice(pos, int(len(pos)*frac), replace=False),
                                   rng.choice(neg, int(len(neg)*frac), replace=False)])
            rng.shuffle(keep); self.pairs = self.pairs[keep]
        self.aug = augment
    def __len__(self): return len(self.pairs)
    def _a(self, x):  # x: (2,32,32,32)
        if not self.aug: return x
        for ax in (1, 2, 3):
            if np.random.rand() < 0.5: x = np.flip(x, ax)
        if np.random.rand() < 0.5: x = np.rot90(x, np.random.randint(1, 4), axes=(2, 3))
        x = x * (1 + (np.random.rand()-0.5)*0.6) + (np.random.rand()-0.5)*0.2  # bright/contrast
        x = x + np.random.randn(*x.shape).astype(np.float32) * 0.01
        return np.ascontiguousarray(np.clip(x, 0, 1))
    def __getitem__(self, k):
        i, j, lbl = self.pairs[k]
        x = np.stack([self.P[i], self.P[j]]).astype(np.float32)
        return torch.from_numpy(self._a(x)), torch.tensor(float(lbl))

class ResBlock(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.c1 = nn.Conv3d(ci, co, 3, stride, 1, bias=False); self.b1 = nn.BatchNorm3d(co)
        self.c2 = nn.Conv3d(co, co, 3, 1, 1, bias=False); self.b2 = nn.BatchNorm3d(co)
        self.sc = nn.Sequential() if (stride == 1 and ci == co) else \
            nn.Sequential(nn.Conv3d(ci, co, 1, stride, bias=False), nn.BatchNorm3d(co))
    def forward(self, x):
        h = F.relu(self.b1(self.c1(x))); h = self.b2(self.c2(h))
        return F.relu(h + self.sc(x))

class ResNet3D(nn.Module):
    def __init__(self):
        super().__init__()
        # stride-2 stem downsamples 32^3 -> 16^3 immediately (8x fewer voxels)
        self.stem = nn.Sequential(nn.Conv3d(2, 24, 3, 2, 1, bias=False), nn.BatchNorm3d(24), nn.ReLU(True))
        self.l1 = nn.Sequential(ResBlock(24, 32, 2))     # 16->8
        self.l2 = nn.Sequential(ResBlock(32, 64, 2))     # 8->4
        self.l3 = nn.Sequential(ResBlock(64, 128, 2))    # 4->2
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.head = nn.Sequential(nn.Linear(128, 96), nn.ReLU(True), nn.Dropout(0.3), nn.Linear(96, 1))
    def forward(self, x):
        h = self.pool(self.l3(self.l2(self.l1(self.stem(x))))).flatten(1)
        return self.head(h).squeeze(1)

def focal(logits, y, a=0.25, g=2.0):
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, y, reduction="none")
    pt = p*y + (1-p)*(1-y); alpha = a*y + (1-a)*(1-y)
    return (alpha * (1-pt).pow(g) * ce).mean()

def load_tensors(files, max_pairs=0, seed=0):
    """Load all patches into one CPU float16 tensor + pairs tensor (no DataLoader)."""
    Ps, prs, off = [], [], 0
    for f in files:
        d = np.load(f); Ps.append(torch.from_numpy(d["patches"]))
        pr = d["pairs"].astype(np.int64).copy(); pr[:, 0] += off; pr[:, 1] += off
        off += len(d["patches"]); prs.append(torch.from_numpy(pr))
    P = torch.cat(Ps); pairs = torch.cat(prs)
    if max_pairs and len(pairs) > max_pairs:
        rng = np.random.default_rng(seed); lbl = pairs[:, 2].numpy()
        pos = np.where(lbl == 1)[0]; neg = np.where(lbl == 0)[0]; frac = max_pairs/len(pairs)
        keep = np.concatenate([rng.choice(pos, int(len(pos)*frac), False),
                               rng.choice(neg, int(len(neg)*frac), False)])
        pairs = pairs[torch.from_numpy(keep)]
    return P, pairs

def augment_gpu(x):  # x (B,2,32,32,32) on GPU
    for ax in (2, 3, 4):
        if np.random.rand() < 0.5: x = torch.flip(x, [ax])
    x = x * (1 + (np.random.rand()-0.5)*0.6) + (np.random.rand()-0.5)*0.2
    return (x + torch.randn_like(x)*0.01).clamp(0, 1)

def run(model, P, pairs, dev, bs, opt=None, aug=False):
    model.train(opt is not None); ys, ps, tot = [], [], 0.0
    idx = torch.randperm(len(pairs)) if opt else torch.arange(len(pairs))
    steps = range(0, len(idx)-(len(idx) % bs if opt else 0), bs)
    for si, b in enumerate(steps):
        if opt and si % 200 == 0: print(f"    step {si}/{len(idx)//bs}", flush=True)
        sel = pairs[idx[b:b+bs]]
        i, j, y = sel[:, 0], sel[:, 1], sel[:, 2].float().to(dev)
        x = torch.stack([P[i], P[j]], 1).float().to(dev)  # (B,2,32,32,32)
        if aug: x = augment_gpu(x)
        with torch.set_grad_enabled(opt is not None):
            lo = model(x); loss = focal(lo, y)
            if opt: opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item()*len(y); ys.append(y.cpu().numpy()); ps.append(torch.sigmoid(lo).detach().cpu().numpy())
    y = np.concatenate(ys); p = np.concatenate(ps)
    return tot/len(y), roc_auc_score(y, p), average_precision_score(y, p)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="/tmp/pairs"); ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--bs", type=int, default=64); ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default="/tmp/student_best.pth")
    ap.add_argument("--max-pairs", type=int, default=0, help="subsample train pairs for fast signal")
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--max-vols", type=int, default=0, help="limit #npz volumes loaded")
    args = ap.parse_args()
    torch.manual_seed(args.seed); np.random.seed(args.seed)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    files = sorted(glob.glob(f"{args.data}/*.npz"))
    if args.max_vols: files = files[:args.max_vols]
    val_files, train_files = files[-2:], files[:-2]
    print(f"train {len(train_files)} vols, val {len(val_files)} vols, dev={dev}, max_pairs={args.max_pairs}", flush=True)
    vmax = args.max_pairs // 4 if args.max_pairs else 0
    t0 = time.time()
    Ptr, prtr = load_tensors(train_files, args.max_pairs, args.seed)
    Pva, prva = load_tensors(val_files, vmax, 1)
    print(f"loaded: train {len(prtr)} pairs, val {len(prva)} pairs in {time.time()-t0:.0f}s", flush=True)
    model = ResNet3D().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.epochs)
    best = 0.0; patience = 0
    for ep in range(args.epochs):
        t = time.time()
        trl, tra, trp = run(model, Ptr, prtr, dev, args.bs, opt=opt, aug=True); sched.step()
        vll, vla, vlp = run(model, Pva, prva, dev, args.bs)
        flag = ""
        if vla > best: best = vla; patience = 0; torch.save(model.state_dict(), args.out); flag = " *BEST*"
        else: patience += 1
        print(f"ep{ep:02d} {time.time()-t:.0f}s trL{trl:.3f} trAUC{tra:.3f} | vlAUC{vla:.4f} vlAP{vlp:.4f}{flag}", flush=True)
        if patience >= 8: print("early stop", flush=True); break
    print(f"BEST val AUC {best:.4f} (seed {args.seed}) -> {args.out}", flush=True)
    print("DONE", flush=True)

if __name__ == "__main__":
    main()
