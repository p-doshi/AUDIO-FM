"""Step 16 - multi-teacher FEATURE distillation vs single-consensus distillation.

The Step-9 model distilled the *averaged* RDM and Step 15 showed it destroys
fine-grained structure. Here: shared student trunk + one linear projector per
teacher, trained so a linear map recovers EACH teacher's actual (z-scored)
embedding - forcing the trunk to hold the union, not the intersection.

Conditions (identical ~130k-param trunk, identical protocol):
  multi19      : 19 per-teacher projector heads, feature-MSE to each
  consensus1   : 1 head, feature-MSE to the mean z-scored teacher embedding
                 (the "averaging" baseline, same trunk)

Then re-run Step 15's within-domain subclass probes on the trunk output and
compare to Step 15's numbers (distilled_phys_mel, phys_frontend, logmel,
foundation).  Output: results/physics_multiteacher.csv
"""
import numpy as np, pandas as pd, torch, torch.nn as nn
import common as C
from step8_phys_arch import CACHE
from step15_subclass_probe import subclass_labels, probe, foundation_embeddings

torch.manual_seed(0); np.random.seed(0)
EPOCHS = 160
BATCH = 128
TRUNK = 256


class Trunk(nn.Module):
    def __init__(self, in_dim, out=TRUNK):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 96, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(96, 128, 5, stride=2, padding=2), nn.ReLU())
        self.head = nn.Linear(128, out)

    def forward(self, x):
        return self.head(self.net(x.transpose(1, 2)).mean(-1))


def train(cond, X, teach_z, dims):
    n = len(X)
    trunk = Trunk(X.shape[2])
    ntrunk = sum(p.numel() for p in trunk.parameters())
    if cond == "multi19":
        projs = nn.ModuleList([nn.Linear(TRUNK, d) for d in dims])
        targets = teach_z                       # list of (n, d_i)
    else:
        projs = nn.ModuleList([nn.Linear(TRUNK, teach_z[0].shape[1] if False else TRUNK)])
        # consensus target: mean over teachers of z-scored embeddings, but dims
        # differ -> project every teacher to a common 256 via PCA-free random? No:
        # use the mean of per-teacher-standardized embeddings truncated/padded.
        common = min(d for d in dims)
        cons = np.mean([t[:, :common] for t in teach_z], axis=0)
        cons = (cons - cons.mean(0)) / (cons.std(0) + 1e-8)
        projs = nn.ModuleList([nn.Linear(TRUNK, common)])
        targets = [cons]
    params = list(trunk.parameters()) + list(projs.parameters())
    opt = torch.optim.Adam(params, lr=1e-3, weight_decay=1e-5)
    Xt = torch.tensor(X)
    Tt = [torch.tensor(t.astype(np.float32)) for t in targets]
    for ep in range(EPOCHS):
        trunk.train()
        perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            opt.zero_grad()
            h = trunk(Xt[idx])
            loss = 0.0
            for pi, tj in zip(projs, Tt):
                pred = pi(h)
                pred = pred / (pred.norm(dim=1, keepdim=True) + 1e-8)
                tgt = tj[idx]
                tgt = tgt / (tgt.norm(dim=1, keepdim=True) + 1e-8)
                loss = loss + ((pred - tgt) ** 2).sum(1).mean()
            (loss / len(Tt)).backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
    trunk.eval()
    with torch.no_grad():
        emb = np.vstack([trunk(Xt[i:i + 256]).numpy() for i in range(0, n, 256)])
    print(f"  [{cond}] trunk params={ntrunk}  emb {emb.shape}")
    return emb


def main():
    d = np.load(CACHE, allow_pickle=True)
    ids = np.array([str(x) for x in d["clip_ids"]]); cats = np.array([str(x) for x in d["categories"]])
    phys, mel = np.nan_to_num(d["phys"]), np.nan_to_num(d["mel"])
    tr = np.arange(len(ids))

    def zn(A):
        f = A[tr].reshape(-1, A.shape[2]); return ((A - f.mean(0)) / (f.std(0) + 1e-6)).astype(np.float32)
    X = np.concatenate([zn(phys), zn(mel)], axis=2)   # (n, 400, 113)
    print(f"[step16] frontend {X.shape}")

    fnd = foundation_embeddings(list(ids))
    dims = [fnd[m].shape[1] for m in C.MODELS]
    teach_z = []
    for m in C.MODELS:
        E = fnd[m]
        teach_z.append(((E - E.mean(0)) / (E.std(0) + 1e-8)).astype(np.float32))

    embs = {}
    for cond in ("consensus1", "multi19"):
        embs[cond] = train(cond, X, teach_z, dims)

    tasks = subclass_labels(ids, cats)
    prev = pd.read_csv(f"{C.OUT}/physics_subclass_probe.csv").set_index("task")
    rows = []
    for task, (sel, y) in tasks.items():
        rec = {"task": task}
        for cond, E in embs.items():
            acc, ncls, nu = probe(E[sel], y)
            rec[f"trunk_{cond}"] = round(acc, 3) if acc == acc else np.nan
        for k in ["distilled_phys_mel", "phys_frontend", "logmel", "foundation_median", "foundation_best"]:
            rec[k] = prev.loc[task, k] if task in prev.index else np.nan
        rows.append(rec)
    df = pd.DataFrame(rows)
    df.to_csv(f"{C.OUT}/physics_multiteacher.csv", index=False)
    pd.set_option("display.width", 240, "display.max_columns", 20)
    print("\n=== Step 16: 5-fold subclass probe accuracy ===")
    print(df.to_string(index=False))
    print(f"\n[step16] saved {C.OUT}/physics_multiteacher.csv")


if __name__ == "__main__":
    main()
