"""Step 8 - a small architecture with the PC1 physics transform as a FIXED
frontend, plus a tiny learned head. Tests how far the shared physical axis
alone carries on the probe data.

Frontend (fixed, no learned params): per STFT frame ->
  [ 16 octave-band group delay , cross-band skew , 48-bin mel-spectrogram
    Laplacian ]  = 65-d sequence, ~43 fps.
Head (learned): 3x Conv1d + global mean pool + linear.  (~80k params)

Conditions (identical head, identical protocol):
  phys      : the 65-d PC1 frontend
  logmel    : 48-d log-mel (generic spectral baseline, matched capacity)
  phys+mel  : 65-d frontend concatenated with 48-d log-mel (does the learned
              semantic part need the raw spectrogram too?)

Task: 8-way probe-set category classification, stratified 70/15/15 split.
Also: RDM of each trained model's pooled embedding on the test set vs
  (a) the Step-7 PC1 fitted RDM, (b) the mean of the 19 foundation-model RDMs.

Outputs: results/physics_arch_results.csv , console report.
"""
import os, sys, time, numpy as np, pandas as pd, librosa, torch
import torch.nn as nn
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import squareform, pdist
import common as C
import phys_frontend as P

CACHE = os.environ.get(
    "PHYS_ARCH_CACHE",
    "/tmp/claude-3160883/-home-user-audio-comp/429fe22d-3649-413e-a9d0-ea529025e0e1/scratchpad/phys_arch_cache.npz")
N_PER_CAT = 375
T_FIX = 400
SEED = 0
EPOCHS = 40
torch.manual_seed(SEED); np.random.seed(SEED)


def pad_fix(a, T=T_FIX):
    if len(a) >= T:
        return a[:T]
    return np.vstack([a, np.zeros((T - len(a), a.shape[1]), a.dtype)])


def build_cache():
    if os.path.exists(CACHE):
        return
    m = pd.read_csv(C.MANIFEST)
    ref = set(np.load(f"{C.EMB_DIR}/clap.npz", allow_pickle=True)["clip_ids"].tolist())
    m = m[m.clip_id.isin(ref) & (m.duration_sec >= 1.0)].copy()
    m["abspath"] = m.path.apply(lambda p: os.path.join(C.DATA_ROOT, p))
    m = m[m.abspath.apply(os.path.exists)]
    rng = np.random.RandomState(SEED)
    parts = []
    for cat, g in m.groupby("category"):
        g = g.sort_values("clip_id")
        parts.append(g.iloc[np.sort(rng.choice(len(g), N_PER_CAT, replace=False))])
    sel = pd.concat(parts).reset_index(drop=True)
    print(f"[step8] extracting frontend for {len(sel)} clips ...")
    phys, mel, cats, ids = [], [], [], []
    t0 = time.time()
    for i, r in sel.iterrows():
        y, _ = librosa.load(r.abspath, sr=P.SR, mono=True, duration=P.MAX_DUR)
        if np.any(y):
            y = librosa.util.normalize(y)
        g16, gsk = P.group_delay_frames(y)
        lap = P.spectral_laplacian_frames(y)
        logmel = librosa.power_to_db(librosa.feature.melspectrogram(
            y=y, sr=P.SR, n_fft=P.N_FFT, hop_length=P.HOP, n_mels=P.N_MELS)).T
        Tn = min(len(g16), len(lap), len(logmel))
        ph = np.concatenate([g16[:Tn], gsk[:Tn, None], lap[:Tn]], 1)
        phys.append(pad_fix(ph.astype(np.float32)))
        mel.append(pad_fix(logmel[:Tn].astype(np.float32)))
        cats.append(r.category); ids.append(r.clip_id)
        if (i + 1) % 400 == 0:
            print(f"   {i+1}/{len(sel)} ({time.time()-t0:.0f}s)")
    np.savez(CACHE, phys=np.array(phys), mel=np.array(mel),
             categories=np.array(cats), clip_ids=np.array(ids))
    print(f"[step8] cached -> {CACHE}  ({time.time()-t0:.0f}s)")


class Head(nn.Module):
    def __init__(self, in_dim, n_cls=8, emb=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, emb, 5, stride=2, padding=2), nn.ReLU(),
        )
        self.fc = nn.Linear(emb, n_cls)

    def forward(self, x):                 # x: (B, T, F)
        h = self.net(x.transpose(1, 2)).mean(-1)   # (B, emb)
        return self.fc(h), h


def run_condition(name, X, y, split, cats_test_ids):
    tr, va, te = split
    dtr = torch.tensor(X[tr]); ytr = torch.tensor(y[tr])
    dva = torch.tensor(X[va]); yva = torch.tensor(y[va])
    dte = torch.tensor(X[te]); yte = torch.tensor(y[te])
    model = Head(X.shape[2])
    npar = sum(p.numel() for p in model.parameters())
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
    lossf = nn.CrossEntropyLoss()
    best_va, best_state = 0, None
    for ep in range(EPOCHS):
        model.train()
        perm = torch.randperm(len(tr))
        for b in range(0, len(tr), 64):
            idx = perm[b:b + 64]
            opt.zero_grad()
            out, _ = model(dtr[idx])
            loss = lossf(out, ytr[idx])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        model.eval()
        with torch.no_grad():
            va_acc = (model(dva)[0].argmax(1) == yva).float().mean().item()
        if va_acc >= best_va:
            best_va, best_state = va_acc, {k: v.clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        logit_te, emb_te = model(dte)
        te_acc = (logit_te.argmax(1) == yte).float().mean().item()
        emb_te = emb_te.numpy()
    per_cls = []
    with torch.no_grad():
        pred = model(dte)[0].argmax(1)
        for c in range(8):
            mc = yte == c
            per_cls.append(round((pred[mc] == c).float().mean().item(), 3))
    print(f"  {name:10s} params={npar:6d}  val_acc={best_va:.3f}  test_acc={te_acc:.3f}")
    return (dict(condition=name, n_params=npar, val_acc=best_va, test_acc=te_acc),
            emb_te, model, per_cls)


STEP1_CACHE = CACHE.replace(".npz", "_step1clips.npz")


def build_step1_cache():
    """Frontend+mel features for the exact 1000 clips used in Step 1, for a
    full-power geometry comparison (inference only, no labels)."""
    if os.path.exists(STEP1_CACHE):
        return
    meta = np.load(f"{C.OUT}/rdm_tensor_meta.npz", allow_pickle=True)
    m = pd.read_csv(C.MANIFEST).set_index("clip_id")
    ids = list(meta["clip_ids"])
    phys, mel = [], []
    print(f"[step8] extracting frontend for the {len(ids)} Step-1 clips ...")
    for i, cid in enumerate(ids):
        ap = os.path.join(C.DATA_ROOT, m.loc[cid, "path"])
        y, _ = librosa.load(ap, sr=P.SR, mono=True, duration=P.MAX_DUR)
        if np.any(y):
            y = librosa.util.normalize(y)
        g16, gsk = P.group_delay_frames(y)
        lap = P.spectral_laplacian_frames(y)
        lm = librosa.power_to_db(librosa.feature.melspectrogram(
            y=y, sr=P.SR, n_fft=P.N_FFT, hop_length=P.HOP, n_mels=P.N_MELS)).T
        Tn = min(len(g16), len(lap), len(lm))
        phys.append(pad_fix(np.concatenate([g16[:Tn], gsk[:Tn, None], lap[:Tn]], 1).astype(np.float32)))
        mel.append(pad_fix(lm[:Tn].astype(np.float32)))
        if (i + 1) % 250 == 0:
            print(f"   {i+1}/{len(ids)}")
    np.savez(STEP1_CACHE, phys=np.nan_to_num(np.array(phys)),
             mel=np.nan_to_num(np.array(mel)), clip_ids=np.array(ids))


def full_geometry(models_by_cond, znorm_stats):
    """Run each trained head over the FULL 1000-clip Step-1 set (inference only)
    and compare the pooled-embedding RDM to: the PC1-fitted RDM, the mean
    19-model RDM, and each of the 19 foundation models individually."""
    build_step1_cache()
    s = np.load(STEP1_CACHE, allow_pickle=True)
    ids1 = list(s["clip_ids"])
    meta = np.load(f"{C.OUT}/rdm_tensor_meta.npz", allow_pickle=True)
    assert ids1 == list(meta["clip_ids"])
    tensor = np.load(f"{C.OUT}/rdm_tensor.npy")
    n = len(ids1); iu = np.triu_indices(n, 1)

    # PC1-fitted RDM on the full set (Step-7 method, mean-19-model target)
    co = list(np.load(f"{C.OUT}/physics_rdms/_clip_order.npy", allow_pickle=True))
    assert co == ids1
    Xg = np.column_stack([rankdata(np.load(f"{C.OUT}/physics_rdms/{g}.npy")[iu])
                          for g in C.FEATURE_GROUPS])
    Xg = (Xg - Xg.mean(0)) / Xg.std(0)
    mean19 = np.stack([rankdata(tensor[:, :, k][iu]) for k in range(19)], 0).mean(0)
    tgt = (mean19 - mean19.mean()) / mean19.std()
    bfit = np.linalg.pinv(Xg.T @ Xg) @ (Xg.T @ tgt)
    pc1_fit = Xg @ bfit

    phys_mu, phys_sd, mel_mu, mel_sd = znorm_stats
    P1 = ((np.nan_to_num(s["phys"]) - phys_mu) / phys_sd).astype(np.float32)
    M1 = ((np.nan_to_num(s["mel"]) - mel_mu) / mel_sd).astype(np.float32)
    feat_for = {"phys": P1, "logmel": M1,
                "phys+mel": np.concatenate([P1, M1], axis=2)}

    rows = []
    for name, model in models_by_cond.items():
        model.eval()
        X = torch.tensor(feat_for[name])
        embs = []
        with torch.no_grad():
            for b0 in range(0, len(X), 128):
                embs.append(model(X[b0:b0 + 128])[1].numpy())
        emb = np.vstack(embs)
        erdm = squareform(pdist(emb, metric="correlation"))[iu]
        r_pc1 = spearmanr(erdm, pc1_fit).statistic
        r_m19 = spearmanr(erdm, mean19).statistic
        permodel = {C.MODELS[k]: round(spearmanr(erdm, tensor[:, :, k][iu]).statistic, 3)
                    for k in range(19)}
        best = max(permodel, key=permodel.get)
        rows.append(dict(condition=name, rsa_vs_pc1_fit=round(r_pc1, 3),
                         rsa_vs_mean19=round(r_m19, 3),
                         best_model=best, best_model_rsa=permodel[best],
                         **{f"rsa_{m}": v for m, v in permodel.items()}))
        print(f"  {name:9s}  RSA vs PC1-fitted={r_pc1:.3f}  vs mean-19={r_m19:.3f}"
              f"  | closest single model: {best} ({permodel[best]:.3f})")
    return rows, pc1_fit, mean19


def main():
    build_cache()
    d = np.load(CACHE, allow_pickle=True)
    d = {k: (np.nan_to_num(d[k]) if k in ("phys", "mel") else d[k]) for k in d.files}
    cats = d["categories"]; ids = d["clip_ids"]
    classes = sorted(set(cats)); cidx = {c: i for i, c in enumerate(classes)}
    y = np.array([cidx[c] for c in cats])
    rng = np.random.RandomState(SEED)
    tr, va, te = [], [], []
    for c in range(len(classes)):
        idx = np.where(y == c)[0]; rng.shuffle(idx)
        n = len(idx); a, b = int(.7 * n), int(.85 * n)
        tr += idx[:a].tolist(); va += idx[a:b].tolist(); te += idx[b:].tolist()
    tr, va, te = map(np.array, (tr, va, te))
    print(f"[step8] split train/val/test = {len(tr)}/{len(va)}/{len(te)}  "
          f"chance = {1/len(classes):.3f}")

    # z-norm each condition's features on train stats
    def stats_of(X):
        flat = X[tr].reshape(-1, X.shape[2])
        return flat.mean(0), flat.std(0) + 1e-6

    phys_mu, phys_sd = stats_of(d["phys"]); mel_mu, mel_sd = stats_of(d["mel"])
    phys = ((d["phys"] - phys_mu) / phys_sd).astype(np.float32)
    mel = ((d["mel"] - mel_mu) / mel_sd).astype(np.float32)
    conds = {"phys": phys, "logmel": mel,
             "phys+mel": np.concatenate([phys, mel], axis=2)}
    classes_l = classes

    res, models_by, per_cls_by = [], {}, {}
    for name, X in conds.items():
        r, emb, model, per_cls = run_condition(name, X, y, (tr, va, te), ids[te])
        res.append(r); models_by[name] = model; per_cls_by[name] = per_cls

    print("\n[step8] per-category test accuracy:")
    print(f"  {'':10s}" + "".join(f"{c[:9]:>11s}" for c in classes_l))
    for name in conds:
        print(f"  {name:10s}" + "".join(f"{v:>11.3f}" for v in per_cls_by[name]))

    print("\n[step8] FULL geometry comparison (1000 Step-1 clips, inference only):")
    grows, pc1_fit, mean19 = full_geometry(
        models_by, (phys_mu, phys_sd, mel_mu, mel_sd))

    out = pd.DataFrame(res).merge(pd.DataFrame(grows), on="condition")
    out.to_csv(f"{C.OUT}/physics_arch_results.csv", index=False)
    core = ["condition", "n_params", "test_acc", "rsa_vs_pc1_fit",
            "rsa_vs_mean19", "best_model", "best_model_rsa"]
    print("\n" + out[core].to_string(index=False))
    # reference: how well does PC1-fit itself track mean-19 (ceiling for our RSA)
    print(f"\n  [ref] PC1-fitted RDM vs mean-19-model RDM: "
          f"{spearmanr(pc1_fit, mean19).statistic:.3f}")
    print(f"[step8] saved {C.OUT}/physics_arch_results.csv")


if __name__ == "__main__":
    main()
