"""Robustness checks on the frozen embeddings of the five public domains (requires the cached embeddings).

Embeddings: $FINETUNE_GEOM_EMB_DIR/<dataset>__<model>.npz (keys embeddings, labels, clip_ids). Output dir: $ROBUST_OUT.

  python robustness_frozen.py probes   kNN with k in {1, 5, 10, 20, 50} under two metrics (Euclidean on standardized
                                       features, as in Finding 2; cosine on raw embeddings), plus the converged linear
                                       probe; accuracy and balanced (macro) accuracy; same splits as finding2_frozen_probes
                                       -> probes_robust.csv
  python robustness_frozen.py geo      uniformity at t in {0.5, 1, 2, 5, 10} (all clips, L2-normalized);
                                       shrinkage H-score (Ibrahim et al., 2022) and GBC (Pandy et al., 2022) on all clips
                                       -> geo_te_robust.csv
  python robustness_frozen.py sim      per domain, 19 x 19 matrices of debiased linear CKA (unbiased HSIC, Song et al.,
                                       2012), standard linear CKA and mutual k-NN overlap (k = 10, cosine)
                                       -> sim_<metric>_<dataset>.csv
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd

from common import DOMAINS
from finding2_frozen_probes import EMB, MODELS, stem, test_mask

OUTD = Path(os.environ.get("ROBUST_OUT", "robust_out"))
OUTD.mkdir(parents=True, exist_ok=True)
KS = [1, 5, 10, 20, 50]
TS = [0.5, 1, 2, 5, 10]


def load(dom, model):
    p = EMB / f"{dom}__{model}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    return z["embeddings"].astype(np.float64), z["labels"].astype(int), z["clip_ids"]


def l2n(X):
    return X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)


# ---------------------------------------------------------------- probes
def probe_job(args):
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    dom, model = args
    d = load(dom, model)
    if d is None:
        return []
    X, y, ids = d
    te = test_mask(dom, ids, y)
    sc = StandardScaler().fit(X[~te])
    feats = {"euclidean_std": (sc.transform(X[~te]), sc.transform(X[te]), "euclidean"),
             "cosine_raw": (l2n(X[~te]), l2n(X[te]), "euclidean")}  # Euclidean on unit vectors == cosine ranking
    rows = []
    for name, (a, b, metric) in feats.items():
        for k in KS:
            p = KNeighborsClassifier(n_neighbors=k, metric=metric, algorithm="brute").fit(a, y[~te]).predict(b)
            rows.append(dict(dataset=dom, model=model, probe="knn", metric=name, k=k,
                             acc=float((p == y[te]).mean()), bal_acc=balanced_accuracy_score(y[te], p)))
    a, b, _ = feats["euclidean_std"]
    p = LogisticRegression(max_iter=10000, tol=1e-6, C=1.0).fit(a, y[~te]).predict(b)
    rows.append(dict(dataset=dom, model=model, probe="linear_conv", metric="euclidean_std", k=np.nan,
                     acc=float((p == y[te]).mean()), bal_acc=balanced_accuracy_score(y[te], p)))
    print(dom, model, "done", flush=True)
    return rows


# ---------------------------------------------------------------- geometry + transferability
def uniformity(Z, t):
    C = Z @ Z.T
    iu = np.triu_indices(len(Z), 1)
    sq = np.clip(2.0 - 2.0 * C[iu], 0, None)
    m = (-t * sq).max()
    return float(np.log(np.mean(np.exp(-t * sq - m))) + m)


def h_alpha(X, y):
    """Shrinkage-based H-score: tr(Sigma_alpha^-1 (1 - alpha) Sigma_between), Ledoit-Wolf shrinkage."""
    from sklearn.covariance import LedoitWolf
    f = X - X.mean(0)
    lw = LedoitWolf(assume_centered=True).fit(f)
    a = lw.shrinkage_
    g = np.zeros_like(f)
    for c in np.unique(y):
        g[y == c] = f[y == c].mean(0)
    cov_g = g.T @ g / len(f)
    return float(np.trace(np.linalg.solve(lw.covariance_, (1 - a) * cov_g)))


def gbc(X, y, dim=64):
    """Gaussian Bhattacharyya coefficient: -sum over class pairs of BC, diagonal Gaussians on PCA features."""
    from sklearn.decomposition import PCA
    Z = PCA(n_components=min(dim, X.shape[1], len(X) - 1), random_state=0).fit_transform(X - X.mean(0))
    cls = np.unique(y)
    mu = np.stack([Z[y == c].mean(0) for c in cls])
    var = np.stack([Z[y == c].var(0) for c in cls]) + 1e-6
    tot = 0.0
    for i in range(len(cls)):
        v = (var[i] + var[i + 1:]) / 2
        db = 0.125 * (((mu[i] - mu[i + 1:]) ** 2) / v).sum(1) + 0.5 * (
            np.log(v).sum(1) - 0.5 * (np.log(var[i]).sum() + np.log(var[i + 1:]).sum(1)))
        tot += np.exp(-db).sum()
    return float(-2 * tot)


def geo_job(args):
    dom, model = args
    d = load(dom, model)
    if d is None:
        return None
    X, y, _ = d
    Z = l2n(X)
    r = dict(dataset=dom, model=model, n=len(y))
    for t in TS:
        r[f"uniformity_t{t:g}"] = uniformity(Z, t)
    r["h_alpha"] = h_alpha(X, y)
    r["gbc"] = gbc(X, y)
    print(dom, model, "done", flush=True)
    return r


# ---------------------------------------------------------------- similarity
def hsic_parts(X):
    X = X - X.mean(0)
    r = (X * X).sum(1)
    s = X.sum(0)
    return X, r, X @ s - r, float(s @ s - r.sum())


def hsic_u(A, B, n):
    Xa, ra, ka, sa = A
    Xb, rb, kb, sb = B
    tr = float(np.sum((Xa.T @ Xb) ** 2) - ra @ rb)
    return (tr + sa * sb / ((n - 1) * (n - 2)) - 2.0 / (n - 2) * float(ka @ kb)) / (n * (n - 3))


def cka_biased(Xa, Xb):
    return float(np.sum((Xa.T @ Xb) ** 2) / np.sqrt(np.sum((Xa.T @ Xa) ** 2) * np.sum((Xb.T @ Xb) ** 2)))


def knn_idx(Z, k=10, chunk=2000):
    Z = Z.astype(np.float32)
    out = np.empty((len(Z), k), np.int32)
    for i in range(0, len(Z), chunk):
        S = Z[i:i + chunk] @ Z.T
        S[np.arange(S.shape[0]), np.arange(i, i + S.shape[0])] = -np.inf
        out[i:i + chunk] = np.argpartition(-S, k, axis=1)[:, :k]
    return out


def sim_domain(dom):
    data = {m: load(dom, m) for m in MODELS}
    data = {m: d for m, d in data.items() if d is not None}
    common_ids = set.intersection(*[set(stem(c) for c in d[2]) for d in data.values()])
    order = sorted(common_ids)
    E = {}
    for m, (X, _, ids) in data.items():
        pos = {stem(c): i for i, c in enumerate(ids)}
        E[m] = X[[pos[c] for c in order]]
    ms = sorted(E)
    n = len(order)
    print(dom, "models", len(ms), "clips", n, flush=True)
    parts = {m: hsic_parts(E[m]) for m in ms}
    self_h = {m: hsic_u(parts[m], parts[m], n) for m in ms}
    nn = {m: knn_idx(l2n(E[m])) for m in ms}
    D, Cb, K = (pd.DataFrame(1.0, index=ms, columns=ms) for _ in range(3))
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            D.loc[a, b] = D.loc[b, a] = hsic_u(parts[a], parts[b], n) / np.sqrt(self_h[a] * self_h[b])
            Cb.loc[a, b] = Cb.loc[b, a] = cka_biased(parts[a][0], parts[b][0])
            K.loc[a, b] = K.loc[b, a] = float((nn[a][:, :, None] == nn[b][:, None, :]).any(-1).sum(1).mean() / 10)
    for name, M in [("dcka", D), ("cka", Cb), ("mknn", K)]:
        M.to_csv(OUTD / f"sim_{name}_{dom}.csv", float_format="%.6f")
    print(dom, "sim done", flush=True)


if __name__ == "__main__":
    task = sys.argv[1]
    nproc = int(os.environ.get("NPROC", "16"))
    jobs = [(d, m) for d in DOMAINS for m in MODELS]
    if task == "probes":
        with Pool(nproc) as p:
            rows = [r for rr in p.imap_unordered(probe_job, jobs) for r in rr]
        pd.DataFrame(rows).sort_values(["dataset", "model", "probe", "metric", "k"]).to_csv(OUTD / "probes_robust.csv", index=False)
    elif task == "geo":
        with Pool(nproc) as p:
            rows = [r for r in p.imap_unordered(geo_job, jobs) if r]
        pd.DataFrame(rows).sort_values(["dataset", "model"]).to_csv(OUTD / "geo_te_robust.csv", index=False)
    elif task == "sim":
        with Pool(min(nproc, len(DOMAINS))) as p:
            p.map(sim_domain, DOMAINS)
