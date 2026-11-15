"""Adaptation gain with an MLP probe head (robustness check for Finding 3; analysis in mlp_head_analysis.py).

Identical to converged_probe_gain.py (same Gram-recovered features, splits, adaptation conditions and checkpoint
filter) except for the classifier: a one-hidden-layer MLP (512 ReLU units, Adam, batch 256, learning rate 1e-3,
at most 300 epochs, early stopping on 10% of the training split, patience 15), with the L2 penalty alpha chosen on
the validation split from {1e-4, 1e-2, 1}, then refit on train and scored on test.
Two feature variants from the same eigendecomposition of the double-centred Gram matrix:
  white  - whitened as in converged_probe_gain.py, divided by sqrt(d) so the MLP sees inputs of unit overall scale;
  scaled - V sqrt(lambda) (the embeddings up to rotation) with one global scale.
Output: one row per model, appended to $MLP_OUT/mlp_gain_<dataset>_seed<seed>.csv; the rows of all calls form
results/robustness/mlp_gain.csv.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.neural_network import MLPClassifier

from common import primary_condition
from converged_probe_gain import CATEGORY_LOADERS, CK, EMB, offset, stem

OUTD = Path(os.environ.get("MLP_OUT", "mlp_out"))
ALPHAS = [1e-4, 1e-2, 1.0]


def features(G, d):
    n = G.shape[0]
    d = min(d, n - 1)
    r = G.mean(1, keepdims=True)
    G = G - r - r.T + G.mean()
    w, V = eigh(G, subset_by_index=[n - d, n - 1])
    w = np.clip(w, 0, None)
    X = V * np.sqrt(w)
    scaled = X / np.sqrt(np.mean(np.sum(X ** 2, 1)) + 1e-12)
    white = V * np.sqrt(w / (w + 1e-3 * w.mean())) * np.sqrt(n)
    return {"scaled": scaled, "white": white / np.sqrt(white.shape[1])}


def mlp(alpha):
    return MLPClassifier(hidden_layer_sizes=(512,), alpha=alpha, batch_size=256, learning_rate_init=1e-3, max_iter=300,
                         early_stopping=True, validation_fraction=0.1, n_iter_no_change=15, random_state=0)


def probe(X, y, itr, iva, ite):
    best = max(ALPHAS, key=lambda a: mlp(a).fit(X[itr], y[itr]).score(X[iva], y[iva]))
    return mlp(best).fit(X[itr], y[itr]).score(X[ite], y[ite]), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    OUTD.mkdir(parents=True, exist_ok=True)
    all_clips, tr, va, _, _, _ = CATEGORY_LOADERS[a.dataset]()
    split = {id(c): "tr" for c in tr}
    split.update({id(c): "va" for c in va})
    rows = []
    for m in a.models:
        cond = primary_condition(m)
        ck = CK / f"{a.dataset}_{cond}_{m}_seed{a.seed}.npz"
        ef = EMB / f"{a.dataset}__{m}.npz"
        if not ck.exists() or not ef.exists():
            continue
        Ga = np.load(ck)["gram"].astype(np.float64)
        z = np.load(ef, allow_pickle=True)
        have = {stem(c) for c in z["clip_ids"]}
        clips = [c for c in all_clips if stem(c["path"]) in have]
        if Ga.shape[0] != len(clips):
            continue
        lut = {stem(c): e for c, e in zip(z["clip_ids"], z["embeddings"])}
        Ef = np.stack([lut[stem(c["path"])] for c in clips]).astype(np.float64)
        xn = Ef / (np.linalg.norm(Ef, axis=1, keepdims=True) + 1e-12)
        if abs(offset(Ga) - float(np.sum(xn.mean(0) ** 2))) < 1e-4:
            continue
        y = np.array([c["label"] for c in clips])
        lab = np.array([split.get(id(c), "te") for c in clips])
        itr, iva, ite = (np.where(lab == s)[0] for s in ("tr", "va", "te"))
        if len(iva) == 0:
            perm = np.random.default_rng(0).permutation(itr)
            nv = int(0.15 * len(itr))
            iva, itr = np.sort(perm[:nv]), np.sort(perm[nv:])
        d = Ef.shape[1]
        F, A = features(Ef @ Ef.T, d), features(Ga, d)
        row = dict(dataset=a.dataset, model=m, condition=cond, seed=a.seed, n_tr=len(itr), n_va=len(iva), n_te=len(ite))
        for tag in ["scaled", "white"]:
            af, alf = probe(F[tag], y, itr, iva, ite)
            aa, ala = probe(A[tag], y, itr, iva, ite)
            row.update({f"frozen_mlp_{tag}": af, f"adapted_mlp_{tag}": aa, f"gain_mlp_{tag}": aa - af,
                        f"alpha_f_{tag}": alf, f"alpha_a_{tag}": ala})
        rows.append(row)
        print(row, flush=True)
    out = OUTD / f"mlp_gain_{a.dataset}_seed{a.seed}.csv"
    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
