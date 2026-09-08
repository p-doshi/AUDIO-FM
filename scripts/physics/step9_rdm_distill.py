"""Step 9 - RDM-distillation: train the small model to REPRODUCE the
representational geometry the 19 foundation models converge on, instead of
predicting categories. Then a linear probe measures transfer.

Target RDM  = mean of the 19 foundation-model rank-RDMs (correlation distance),
              computed directly from the cached .npz embeddings.
Loss (per batch of B clips):
     D_pred = 1 - corr(emb_i, emb_j)            (B x B, differentiable)
     L = (1 - pearson(uppertri D_pred, uppertri D_tgt))  +  0.1 * MSE(z,z)
Frontends: phys (PC1 transform) | logmel | phys+mel  -- identical head,
identical protocol.  Head outputs a 128-d embedding, NO classifier.

Eval on held-out clips (Step-1 1000-set minus any training clip):
  * RSA of learned-embedding RDM vs mean-19 consensus, vs PC1-fitted RDM,
    vs each of the 19 models
  * linear-probe (logistic regression on frozen embeddings) category accuracy
Compare against the Step-8 category-supervised numbers.

Outputs: results/physics_distill_results.csv
"""
import os, time, numpy as np, pandas as pd, torch
import torch.nn as nn
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import common as C
import phys_frontend as P
from step8_phys_arch import CACHE, STEP1_CACHE, build_cache, build_step1_cache, pad_fix, Head

SEED = 0
EPOCHS = 120
BATCH = 256
EMB = 128
torch.manual_seed(SEED); np.random.seed(SEED)
OUT_CSV = f"{C.OUT}/physics_distill_results.csv"


class Embed(nn.Module):
    """Same conv stack as Step-8 Head, but returns only the 128-d embedding."""
    def __init__(self, in_dim, emb=EMB):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, emb, 5, stride=2, padding=2), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).mean(-1)


def mean19_rank_rdm(clip_ids):
    """(n,n) mean of 19 foundation-model rank RDMs (correlation distance)."""
    want = list(clip_ids)
    n = len(want)
    iu = np.triu_indices(n, 1)
    acc = np.zeros(len(iu[0]))
    for mdl in C.MODELS:
        d = np.load(f"{C.EMB_DIR}/{mdl}.npz", allow_pickle=True)
        pos = {c: i for i, c in enumerate(d["clip_ids"].tolist())}
        E = d["embeddings"][[pos[c] for c in want]].astype(np.float64)
        rdm = squareform(pdist(E, metric="correlation"))[iu]
        acc += rankdata(rdm)
    acc /= len(C.MODELS)
    full = np.zeros((n, n)); full[iu] = acc; full = full + full.T
    return full


def batch_rdm_loss(emb, tgt_ut, iu):
    e = emb - emb.mean(0, keepdim=True)
    e = e / (e.norm(dim=1, keepdim=True) + 1e-8)
    corr = e @ e.t()
    d = 1.0 - corr
    pred = d[iu[0], iu[1]]
    pz = (pred - pred.mean()) / (pred.std() + 1e-8)
    tz = (tgt_ut - tgt_ut.mean()) / (tgt_ut.std() + 1e-8)
    pearson = (pz * tz).mean()
    return (1 - pearson) + 0.1 * ((pz - tz) ** 2).mean()


def train_distill(name, Xtr, tgt_tr, Xva, tgt_va):
    model = Embed(Xtr.shape[2])
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    Xtr_t = torch.tensor(Xtr); Xva_t = torch.tensor(Xva)
    tgt_tr_t = torch.tensor(tgt_tr.astype(np.float32))
    n = len(Xtr)
    va_iu = np.triu_indices(len(Xva), 1)
    tgt_va_ut = tgt_va[va_iu]
    best, best_state = -1, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(n)
        for b in range(0, n, BATCH):
            idx = perm[b:b + BATCH]
            if len(idx) < 16:
                continue
            iu = torch.triu_indices(len(idx), len(idx), 1)
            sub = tgt_tr_t[idx][:, idx]
            tgt_ut = sub[iu[0], iu[1]]
            opt.zero_grad()
            emb = model(Xtr_t[idx])
            loss = batch_rdm_loss(emb, tgt_ut, iu)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            ev = model(Xva_t).numpy()
        erdm = squareform(pdist(ev, metric="correlation"))[va_iu]
        rsa = spearmanr(erdm, tgt_va_ut).statistic
        if rsa > best:
            best, best_state = rsa, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    print(f"  {name:9s} params={npar}  best val RSA-to-consensus={best:.3f}")
    return model, npar


def embed_all(model, X):
    model.eval()
    out = []
    with torch.no_grad():
        for b in range(0, len(X), 256):
            out.append(model(torch.tensor(X[b:b + 256])).numpy())
    return np.vstack(out)


def main():
    build_cache(); build_step1_cache()
    d = np.load(CACHE, allow_pickle=True)
    tr_ids = list(d["clip_ids"]); tr_cats = d["categories"]
    phys_raw, mel_raw = np.nan_to_num(d["phys"]), np.nan_to_num(d["mel"])

    s = np.load(STEP1_CACHE, allow_pickle=True)
    s1_ids = list(s["clip_ids"])
    s1_phys, s1_mel = np.nan_to_num(s["phys"]), np.nan_to_num(s["mel"])

    # held-out = Step-1 clips not in the distillation training pool
    train_set = set(tr_ids)
    held = [i for i, c in enumerate(s1_ids) if c not in train_set]
    print(f"[step9] train pool={len(tr_ids)}  Step-1 held-out (no overlap)={len(held)}")
    held_ids = [s1_ids[i] for i in held]

    # splits within the training pool
    rng = np.random.RandomState(SEED)
    idx = rng.permutation(len(tr_ids))
    va_n = 400
    va_idx, fit_idx = idx[:va_n], idx[va_n:]

    # targets
    print("[step9] building mean-19 consensus RDM on the training pool ...")
    tgt_full = mean19_rank_rdm(tr_ids)
    tgt_fit = tgt_full[np.ix_(fit_idx, fit_idx)]
    tgt_va = tgt_full[np.ix_(va_idx, va_idx)]

    # held-out reference RDMs
    meta = np.load(f"{C.OUT}/rdm_tensor_meta.npz", allow_pickle=True)
    tensor = np.load(f"{C.OUT}/rdm_tensor.npy")
    tp = {c: i for i, c in enumerate(meta["clip_ids"])}
    hk = np.array([tp[c] for c in held_ids])
    hiu = np.triu_indices(len(hk), 1)
    held_mean19 = np.stack([rankdata(tensor[np.ix_(hk, hk, [k])][:, :, 0][hiu])
                            for k in range(19)], 0).mean(0)
    co = list(np.load(f"{C.OUT}/physics_rdms/_clip_order.npy", allow_pickle=True))
    ck = np.array([co.index(c) for c in held_ids])
    Xg = np.column_stack([rankdata(np.load(f"{C.OUT}/physics_rdms/{g}.npy")[np.ix_(ck, ck)][hiu])
                          for g in C.FEATURE_GROUPS])
    Xg = (Xg - Xg.mean(0)) / Xg.std(0)
    tz = (held_mean19 - held_mean19.mean()) / held_mean19.std()
    pc1_fit_held = Xg @ (np.linalg.pinv(Xg.T @ Xg) @ (Xg.T @ tz))
    print(f"  [ref] held-out PC1-fit vs mean-19 = {spearmanr(pc1_fit_held, held_mean19).statistic:.3f}")

    def zn(fit_arr, *arrs):
        flat = fit_arr.reshape(-1, fit_arr.shape[2])
        mu, sd = flat.mean(0), flat.std(0) + 1e-6
        return [((a - mu) / sd).astype(np.float32) for a in (fit_arr, *arrs)]

    ph_fit, ph_all, ph_held = zn(phys_raw[fit_idx], phys_raw, s1_phys[held])
    me_fit, me_all, me_held = zn(mel_raw[fit_idx], mel_raw, s1_mel[held])
    ph_va = ph_all[va_idx]; me_va = me_all[va_idx]

    fe = {
        "phys":     (ph_fit, ph_va, ph_held),
        "logmel":   (me_fit, me_va, me_held),
        "phys+mel": (np.concatenate([ph_fit, me_fit], 2),
                     np.concatenate([ph_va, me_va], 2),
                     np.concatenate([ph_held, me_held], 2)),
    }
    # probe labels
    cl = sorted(set(tr_cats)); ci = {c: i for i, c in enumerate(cl)}
    y_fit = np.array([ci[tr_cats[i]] for i in fit_idx])
    mani = pd.read_csv(C.MANIFEST).set_index("clip_id")["category"]
    y_held = np.array([ci[mani[c]] for c in held_ids])

    # z-norm stats needed to reproduce the frontend at inference (vessel script)
    def stats_of(a):
        f = a.reshape(-1, a.shape[2]); return f.mean(0), f.std(0) + 1e-6
    znstats = {"phys": stats_of(phys_raw[fit_idx]), "mel": stats_of(mel_raw[fit_idx])}

    rows = []
    for name, (Xf, Xv, Xh) in fe.items():
        model, npar = train_distill(name, Xf, tgt_fit, Xv, tgt_va)
        ck = {
            "state_dict": model.state_dict(), "in_dim": Xf.shape[2],
            "emb": EMB, "condition": name,
            "frontend": {"SR": P.SR, "N_FFT": P.N_FFT, "HOP": P.HOP,
                         "N_MELS": P.N_MELS, "N_GD_BANDS": P.N_GD_BANDS,
                         "T_FIX": 400, "MAX_DUR": P.MAX_DUR},
            "znorm": {"phys": tuple(map(lambda x: x.tolist(), znstats["phys"])),
                      "mel": tuple(map(lambda x: x.tolist(), znstats["mel"]))},
        }
        torch.save(ck, f"{C.OUT}/physics_distill_ckpt_{name.replace('+','_')}.pt")
        eh = embed_all(model, Xh)
        erdm = squareform(pdist(eh, metric="correlation"))[hiu]
        rsa_cons = spearmanr(erdm, held_mean19).statistic
        rsa_pc1 = spearmanr(erdm, pc1_fit_held).statistic
        permodel = {C.MODELS[k]: round(spearmanr(erdm, tensor[np.ix_(hk, hk, [k])][:, :, 0][hiu]).statistic, 3)
                    for k in range(19)}
        best = max(permodel, key=permodel.get)
        # linear probe
        ef = embed_all(model, Xf)
        sc = StandardScaler().fit(ef)
        clf = LogisticRegression(max_iter=2000, C=1.0).fit(sc.transform(ef), y_fit)
        probe_acc = clf.score(sc.transform(eh), y_held)
        rows.append(dict(condition=name, n_params=npar,
                         held_RSA_vs_consensus=round(rsa_cons, 3),
                         held_RSA_vs_PC1fit=round(rsa_pc1, 3),
                         closest_model=best, closest_model_RSA=permodel[best],
                         linear_probe_acc=round(probe_acc, 3),
                         **{f"rsa_{m}": v for m, v in permodel.items()}))
        print(f"  {name:9s} held-out: RSA vs consensus={rsa_cons:.3f}  "
              f"vs PC1-fit={rsa_pc1:.3f}  closest={best}({permodel[best]:.3f})  "
              f"linear-probe acc={probe_acc:.3f}")

    df = pd.DataFrame(rows)
    df.to_csv(OUT_CSV, index=False)
    core = ["condition", "n_params", "held_RSA_vs_consensus", "held_RSA_vs_PC1fit",
            "closest_model", "closest_model_RSA", "linear_probe_acc"]
    print("\n" + df[core].to_string(index=False))
    print(f"\n[step9] Step-8 reference (category-supervised): "
          f"phys probe-equiv 0.941 / geom-vs-consensus 0.364; "
          f"phys+mel 0.961 / 0.501")
    print(f"[step9] saved {OUT_CSV}")


if __name__ == "__main__":
    main()
