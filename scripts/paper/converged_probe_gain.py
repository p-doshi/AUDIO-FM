"""Converged linear probe on the frozen and on the adapted encoder, and adaptation gain.

Inputs
  adapted: the Gram matrix E E^T over the domain's full clip pool (train + val + test, loader order), stored
           per (dataset, condition, model, seed) by audio_comp.pipelines.rsa_cka_finetuned in
           $RSA_CKA_CHECKPOINT_DIR/<dataset>_<condition>_<model>_seed<seed>.npz
  frozen:  cached frozen embeddings, $FINETUNE_GEOM_EMB_DIR/<dataset>__<model>.npz
Both sides go through the same route: double-centre the Gram matrix, take its top-d eigenpairs (d = the
encoder's embedding width), and whiten every principal direction to about unit variance with a small ridge
(eps = 1e-3 x mean eigenvalue). This recovers the embeddings up to rotation; whitening is the
rotation-invariant counterpart of per-feature standardization.
Probe: L2 logistic regression (L-BFGS, max_iter 3000), C chosen on the validation split from
{1e-2, 1e-1, 1, 10, 1e2, 1e3}, refit on train, scored on test. The machine loader has no validation split,
so a fixed 15% of its training pool (default_rng(0)) is held out to choose C.
gain = adapted accuracy - frozen accuracy.
Adapted checkpoints that are numerically identical to the frozen encoder (shared offset within 1e-4 of the
frozen value) are skipped.
Condition: --cond primary = LoRA for the 14 LoRA-eligible models and top-K for the other 4; or --cond allora.
Output: results/probes/converged/converged_probe_gain_<dataset>[_allora]_seed<seed>.csv
"""
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.linalg import eigh
from sklearn.linear_model import LogisticRegression

from common import REPO, RESULTS, primary_condition

sys.path.insert(0, str(REPO))
from audio_comp.pipelines.rsa_cka_finetuned import CATEGORY_LOADERS  # noqa: E402

CK = Path(os.environ.get("RSA_CKA_CHECKPOINT_DIR", REPO / "audio_comp" / "pipelines" / ".rsa_finetuned_checkpoints"))
EMB = Path(os.environ.get("FINETUNE_GEOM_EMB_DIR", REPO / "results" / "finetune_data_geometry_emb"))
CS = [1e-2, 1e-1, 1.0, 10.0, 100.0, 1000.0]


def stem(p):
    """Clip key. Machine (MIMII) file names repeat across machine / unit / condition, so key on the last 4 path parts."""
    p = str(p)
    return "/".join(os.path.splitext(p)[0].split("/")[-4:]) if "mimii_raw" in p else os.path.splitext(os.path.basename(p))[0]


def features(G, d):
    n = G.shape[0]
    d = min(d, n - 1)
    r = G.mean(1, keepdims=True)
    G = G - r - r.T + G.mean()
    w, V = eigh(G, subset_by_index=[n - d, n - 1])
    w = np.clip(w, 0, None)
    return V * np.sqrt(w / (w + 1e-3 * w.mean())) * np.sqrt(n)


def offset(G):
    s = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    return float((G / s[:, None] / s[None]).mean())


def probe(X, y, itr, iva, ite):
    best = max(CS, key=lambda C: LogisticRegression(C=C, max_iter=3000).fit(X[itr], y[itr]).score(X[iva], y[iva]))
    return LogisticRegression(C=best, max_iter=3000).fit(X[itr], y[itr]).score(X[ite], y[ite]), best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cond", default="primary", choices=["primary", "allora"])
    a = ap.parse_args()
    all_clips, tr, va, _, _, _ = CATEGORY_LOADERS[a.dataset]()
    split = {id(c): "tr" for c in tr}
    split.update({id(c): "va" for c in va})
    rows = []
    for m in a.models:
        cond = primary_condition(m) if a.cond == "primary" else "allora"
        ck = CK / f"{a.dataset}_{cond}_{m}_seed{a.seed}.npz"
        ef = EMB / f"{a.dataset}__{m}.npz"
        if not ck.exists() or not ef.exists():
            print("missing", ck if not ck.exists() else ef, flush=True)
            continue
        Ga = np.load(ck)["gram"].astype(np.float64)
        z = np.load(ef, allow_pickle=True)
        have = {stem(c) for c in z["clip_ids"]}
        clips = [c for c in all_clips if stem(c["path"]) in have]
        if Ga.shape[0] != len(clips):
            print(f"{m}: gram n={Ga.shape[0]} != aligned clips {len(clips)}; skipping", flush=True)
            continue
        lut = {stem(c): e for c, e in zip(z["clip_ids"], z["embeddings"])}
        Ef = np.stack([lut[stem(c["path"])] for c in clips]).astype(np.float64)
        xn = Ef / (np.linalg.norm(Ef, axis=1, keepdims=True) + 1e-12)
        if abs(offset(Ga) - float(np.sum(xn.mean(0) ** 2))) < 1e-4:
            print(f"{m}: adapted checkpoint equals the frozen encoder; skipping", flush=True)
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
        af, cf = probe(F, y, itr, iva, ite)
        aa, ca = probe(A, y, itr, iva, ite)
        rows.append(dict(dataset=a.dataset, model=m, condition=cond, seed=a.seed, n_tr=len(itr), n_va=len(iva), n_te=len(ite),
                         frozen_white=af, adapted_white=aa, gain_white=aa - af, C_f_white=cf, C_a_white=ca))
        print(rows[-1], flush=True)
    tag = "" if a.cond == "primary" else "_allora"
    out = RESULTS / "probes" / "converged" / f"converged_probe_gain_{a.dataset}{tag}_seed{a.seed}.csv"
    pd.DataFrame(rows).to_csv(out, mode="a", header=not out.exists(), index=False)


if __name__ == "__main__":
    main()
