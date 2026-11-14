"""Layer-wise analyses on frozen embeddings captured from every encoder block (five public domains, 18 models;
EnCodecMAE's adapter does not expose its blocks).

Input: $LAYER_EMB_DIR/<dataset>__<model>.npz from layerwise_extract.py (layer_<i>, clip_ids, labels).
Per (dataset, model, layer), using the Finding 2 splits (finding2_frozen_probes.test_mask):
  uniformity (t = 2, all clip pairs, L2-normalized), kNN accuracy (k = 10, Euclidean on standardized features),
  linear accuracy (logistic regression, C = 1, max_iter 500, tol 1e-3).
  -> $ROBUST_OUT/layerwise_probes.csv
Cross-domain stability at relative depth 0.5 (block nearest the middle): 18 x 18 RSA (Spearman between correlation-
distance RDMs) and debiased linear CKA per domain -> $ROBUST_OUT/layer_mid_{rsa,dcka}_<dataset>.csv
Layers with fewer than 16 dimensions (PANNs pre-processing stages) are skipped.
"""
import os
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from common import DOMAINS
from finding2_frozen_probes import MODELS, stem, test_mask
from robustness_frozen import hsic_parts, hsic_u, l2n, uniformity

LAY = Path(os.environ["LAYER_EMB_DIR"])
OUTD = Path(os.environ.get("ROBUST_OUT", "robust_out"))
OUTD.mkdir(parents=True, exist_ok=True)
LMODELS = [m for m in MODELS if m != "encodecmae"]


def layers_of(z):
    return [k for k in sorted(z.files, key=lambda s: int(s.split("_")[1]) if s.startswith("layer_") else -1)
            if k.startswith("layer_") and z[k].shape[1] >= 16]


def probe_job(args):
    dom, model = args
    p = LAY / f"{dom}__{model}.npz"
    if not p.exists():
        return []
    z = np.load(p, allow_pickle=True)
    y = z["labels"].astype(int)
    te = test_mask(dom, z["clip_ids"], y)
    ks = layers_of(z)
    rows = []
    for j, k in enumerate(ks):
        X = z[k].astype(np.float64)
        sc = StandardScaler().fit(X[~te])
        a, b = sc.transform(X[~te]), sc.transform(X[te])
        knn = KNeighborsClassifier(n_neighbors=10, algorithm="brute").fit(a, y[~te]).score(b, y[te])
        lin = LogisticRegression(max_iter=500, tol=1e-3, C=1.0).fit(a, y[~te]).score(b, y[te])
        rows.append(dict(dataset=dom, model=model, layer=j, n_layers=len(ks), depth=j / max(len(ks) - 1, 1),
                         dim=X.shape[1], uniformity=uniformity(l2n(X), 2.0), knn=knn, linear=lin))
    print(dom, model, len(ks), "layers", flush=True)
    return rows


def mid_layer(dom):
    raw, ms = {}, []
    for m in LMODELS:
        p = LAY / f"{dom}__{m}.npz"
        if not p.exists():
            continue
        z = np.load(p, allow_pickle=True)
        ks = layers_of(z)
        raw[m] = ([str(c) for c in z["clip_ids"]], z[ks[int(round(0.5 * (len(ks) - 1)))]].astype(np.float64))
        ms.append(m)
    order = sorted(set.intersection(*[set(i) for i, _ in raw.values()]))
    E = {}
    for m, (ids, X) in raw.items():
        pos = {c: i for i, c in enumerate(ids)}
        E[m] = X[[pos[c] for c in order]]
    n = len(order)
    parts = {m: hsic_parts(E[m]) for m in ms}
    sh = {m: hsic_u(parts[m], parts[m], n) for m in ms}
    ranks = {m: rankdata(pdist(E[m], "correlation")).astype(np.float32) for m in ms}
    for m in ms:
        ranks[m] -= ranks[m].mean()
        ranks[m] /= np.linalg.norm(ranks[m])
    R, D = (pd.DataFrame(1.0, index=ms, columns=ms) for _ in range(2))
    for i, a in enumerate(ms):
        for b in ms[i + 1:]:
            R.loc[a, b] = R.loc[b, a] = float(ranks[a].astype(np.float64) @ ranks[b].astype(np.float64))
            D.loc[a, b] = D.loc[b, a] = hsic_u(parts[a], parts[b], n) / np.sqrt(sh[a] * sh[b])
    R.to_csv(OUTD / f"layer_mid_rsa_{dom}.csv", float_format="%.6f")
    D.to_csv(OUTD / f"layer_mid_dcka_{dom}.csv", float_format="%.6f")
    print(dom, "mid-layer similarity done", flush=True)


if __name__ == "__main__":
    task = sys.argv[1]
    nproc = int(os.environ.get("NPROC", "16"))
    if task == "probes":
        with Pool(nproc) as p:
            rows = [r for rr in p.imap_unordered(probe_job, [(d, m) for d in DOMAINS for m in LMODELS]) for r in rr]
        pd.DataFrame(rows).sort_values(["dataset", "model", "layer"]).to_csv(OUTD / "layerwise_probes.csv", index=False)
    elif task == "sim":
        for d in DOMAINS:
            mid_layer(d)
