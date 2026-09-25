"""SELF-SUPERVISED decisive experiment: SimCLR contrastive pretraining on the
UNLABELED cell patches, then evaluate whether the learned embeddings separate
REAL-GT edges from non-edges better than the geometric baseline (AUC 0.849) and
the supervised appearance encoder (0.828). Uses data structure, NOT pilkwang's
labels -> the only route with headroom above the distillation ceiling.

Corpus: patches from /tmp/pairs (unlabeled detections). Eval: /tmp/contrarian
pairs (labeled by real sparse GT). Decisive: cosine-sim AUC on real GT.
"""
import sys, glob, time, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
sys.path.insert(0, "src")
from train_student import ResBlock
from sklearn.metrics import roc_auc_score

dev = "cuda"

class Encoder(nn.Module):
    def __init__(self, dim=128, proj=64):
        super().__init__()
        self.stem = nn.Sequential(nn.Conv3d(1, 24, 3, 2, 1, bias=False), nn.BatchNorm3d(24), nn.ReLU(True))
        self.l1 = ResBlock(24, 32, 2); self.l2 = ResBlock(32, 64, 2); self.l3 = ResBlock(64, dim, 2)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.proj = nn.Sequential(nn.Linear(dim, dim), nn.ReLU(True), nn.Linear(dim, proj))
    def features(self, x):
        return self.pool(self.l3(self.l2(self.l1(self.stem(x))))).flatten(1)
    def forward(self, x):
        return F.normalize(self.proj(self.features(x)), dim=1)

def aug(x):  # x (B,1,32,32,32) on GPU -> one augmented view
    for ax in (2, 3, 4):
        if np.random.rand() < 0.5: x = torch.flip(x, [ax])
    if np.random.rand() < 0.5: x = torch.rot90(x, np.random.randint(1, 4), dims=(3, 4))
    x = x * (1 + (np.random.rand()-0.5)*0.8) + (np.random.rand()-0.5)*0.3
    return (x + torch.randn_like(x)*0.02).clamp(0, 1)

def nt_xent(z1, z2, tau=0.2):
    N = z1.shape[0]; z = torch.cat([z1, z2], 0)
    sim = (z @ z.t()) / tau
    sim.fill_diagonal_(-9e9)
    targets = torch.cat([torch.arange(N, 2*N), torch.arange(0, N)]).to(z.device)
    return F.cross_entropy(sim, targets)

def load_patches(files, cap):
    Ps = []
    for f in files:
        Ps.append(np.load(f)["patches"])
        if sum(len(p) for p in Ps) > cap: break
    P = np.concatenate(Ps)[:cap]
    return torch.from_numpy(P)  # (N,32,32,32) float16 CPU

# ---- unlabeled corpus ----
t0 = time.time()
corpus = load_patches(sorted(glob.glob("/tmp/pairs/*.npz")), cap=200000)
print(f"corpus {len(corpus)} unlabeled patches in {time.time()-t0:.0f}s", flush=True)
m = Encoder().to(dev)
opt = torch.optim.AdamW(m.parameters(), lr=3e-4, weight_decay=1e-5)
sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, 40)
bs = 256

# ---- eval set (real GT) ----
def load_eval():
    Ps, prs, off = [], [], 0
    for f in sorted(glob.glob("/tmp/contrarian/*.npz")):
        d = np.load(f); Ps.append(d["patches"])
        pr = d["pairs"].astype(np.int64).copy(); pr[:, 0] += off; pr[:, 1] += off
        off += len(d["patches"]); prs.append(pr)
    return torch.from_numpy(np.concatenate(Ps)), np.concatenate(prs)
Pev, prev = load_eval()
print(f"eval {len(prev)} real-GT pairs ({int((prev[:,2]==1).sum())}+/{int((prev[:,2]==0).sum())}-)", flush=True)

def evaluate():
    m.eval()
    with torch.no_grad():
        embs = []
        for b in range(0, len(Pev), 512):
            x = Pev[b:b+512].unsqueeze(1).float().to(dev)
            embs.append(F.normalize(m.features(x), dim=1).cpu())
        E = torch.cat(embs)
        i, j, y = prev[:, 0], prev[:, 1], prev[:, 2].astype(float)
        cos = (E[i] * E[j]).sum(1).numpy()
    auc = roc_auc_score(y, cos)
    neg = y == 0
    # separability of hard negatives (pilkwang-contradicting, nearby diff cells)
    hardneg_auc = roc_auc_score(np.r_[np.ones(3), np.zeros(neg.sum())],
                                np.r_[cos[y==1][:3], cos[neg]]) if neg.sum() else float("nan")
    return auc

print(f"BEFORE (random init) real-GT cos AUC: {evaluate():.4f}", flush=True)
for ep in range(40):
    m.train(); idx = torch.randperm(len(corpus)); tot = 0.0; nb = 0
    for b in range(0, len(corpus)-bs, bs):
        x = corpus[idx[b:b+bs]].unsqueeze(1).float().to(dev)
        z1, z2 = m(aug(x)), m(aug(x))
        loss = nt_xent(z1, z2); opt.zero_grad(); loss.backward(); opt.step()
        tot += loss.item(); nb += 1
    sched.step()
    if ep % 4 == 3 or ep == 0:
        print(f"ep{ep:02d} ntxent {tot/nb:.3f} | real-GT cos AUC {evaluate():.4f}", flush=True)
torch.save(m.state_dict(), "/tmp/simclr_encoder.pth")
print(f"FINAL real-GT cos AUC {evaluate():.4f}  (baseline geometric 0.849, appearance 0.828)", flush=True)
print("DONE", flush=True)
