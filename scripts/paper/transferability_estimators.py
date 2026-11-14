"""Label-based transferability estimators on frozen embeddings (Finding 3 baselines).

For every (dataset, model) with cached frozen embeddings ($FINETUNE_GEOM_EMB_DIR/<dataset>__<model>.npz):
  knn_ref_accuracy  kNN accuracy (k = 5, cosine similarity) using up to 100 labelled reference clips per class
                    (at most half of each class; default_rng(0)), evaluated on the rest of the domain's pool
  logme             LogME (You et al., 2021): mean over classes of the per-sample log evidence of a Bayesian
                    linear model of the one-vs-rest +-1 target on the frozen features
Output: results/probes/transferability_estimators.csv (uniformity added from results/geometry/frozen_geometry.csv)
"""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import pandas as pd

from common import DOMAINS, RESULTS, frozen_geometry
from finding2_frozen_probes import EMB, MODELS

K_REF_PER_CLASS = 100
KNN_K = 5
SEED = 0


def stratified_ref_test_split(labels, k_per_class, seed):
    rng = np.random.default_rng(seed)
    labels = np.asarray(labels)
    ref_idx, test_idx = [], []
    for c in np.unique(labels):
        idx = np.where(labels == c)[0]
        rng.shuffle(idx)
        n_ref = min(k_per_class, max(1, len(idx) // 2))
        ref_idx.extend(idx[:n_ref])
        test_idx.extend(idx[n_ref:])
    return np.array(ref_idx), np.array(test_idx)


def knn_accuracy(emb, labels, ref_idx, test_idx, k=5):
    ref = emb[ref_idx]
    ref_lab = labels[ref_idx]
    test = emb[test_idx]
    test_lab = labels[test_idx]
    ref_n = ref / (np.linalg.norm(ref, axis=1, keepdims=True) + 1e-12)
    test_n = test / (np.linalg.norm(test, axis=1, keepdims=True) + 1e-12)
    sims = test_n @ ref_n.T
    correct = 0
    for i in range(test.shape[0]):
        top_k = np.argpartition(-sims[i], min(k, sims.shape[1] - 1))[:k]
        votes = ref_lab[top_k]
        vals, counts = np.unique(votes, return_counts=True)
        pred = vals[np.argmax(counts)]
        correct += int(pred == test_lab[i])
    return correct / test.shape[0]


def logme(features, labels):
    f = features.astype(np.float64)
    n, d = f.shape
    labels = np.asarray(labels)
    classes = np.unique(labels)
    u, s, vh = np.linalg.svd(f, full_matrices=False)
    sigma = s**2

    scores = []
    for c in classes:
        y = (labels == c).astype(np.float64) * 2 - 1
        z = u.T @ y
        alpha, beta = 1.0, 1.0
        for _ in range(30):
            w_coef = (beta * sigma * z) / (alpha + beta * sigma)
            m_norm2 = np.sum(w_coef**2)
            gamma2 = np.sum(beta * sigma / (alpha + beta * sigma))
            alpha_new = gamma2 / (m_norm2 + 1e-12)
            resid = np.sum(y**2) - np.sum((beta * sigma * z**2) / (alpha + beta * sigma))
            beta_new = (n - gamma2) / (resid + 1e-12)
            if abs(alpha_new - alpha) < 1e-4 and abs(beta_new - beta) < 1e-4:
                alpha, beta = alpha_new, beta_new
                break
            alpha, beta = alpha_new, beta_new
        alpha = max(alpha, 1e-8)
        beta = max(beta, 1e-8)
        evidence = (
            0.5 * d * np.log(alpha)
            + 0.5 * n * np.log(beta)
            - 0.5 * np.sum(np.log(alpha + beta * sigma))
            - 0.5 * beta * (np.sum(y**2) - np.sum((beta * sigma * z**2) / (alpha + beta * sigma)))
            - 0.5 * n * np.log(2 * np.pi)
        )
        scores.append(evidence / n)
    return float(np.mean(scores))


def main():
    uni = {(r.dataset, r.model): r.uniformity for r in frozen_geometry().itertuples()}
    rows = []
    for dom in DOMAINS:
        for m in MODELS:
            p = EMB / f"{dom}__{m}.npz"
            if not p.exists() or (dom, m) not in uni:
                continue
            z = np.load(p, allow_pickle=True)
            emb, labels = z["embeddings"].astype(np.float64), z["labels"]
            ref, test = stratified_ref_test_split(labels, K_REF_PER_CLASS, SEED)
            rows.append(dict(domain=dom, model=m, n_clips=len(emb),
                             knn_ref_accuracy=knn_accuracy(emb, labels, ref, test, k=KNN_K),
                             logme=logme(emb, labels), uniformity=uni[(dom, m)]))
            print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv(RESULTS / "probes" / "transferability_estimators.csv", index=False)


if __name__ == "__main__":
    main()
