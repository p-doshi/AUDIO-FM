"""Three further transferability estimators on the frozen embeddings of the five public domains (all clips and labels).

  sfda      SFDA (Shao et al., ECCV 2022): self-challenging Fisher discriminant analysis; official implementation
            (github.com/TencentARC/SFDA, metrics.py: SFDA, SFDA_Score), reproduced below without logging.
  ncti      NCTI (Wang et al., ICCV 2023): official implementation (github.com/BUserName/NCTI, metrics.py: NCTI_Score)
            gives three per-model terms on 64 PCA dimensions: nuclear norm of the centred LDA projection (all_nuc),
            nearest-class-centre posterior of the shrinkage LDA (ncc) and log of the summed per-class nuclear norm of
            the posteriors (cls_nuc). As in the official evaluation script, the score is min-max(ncc) + min-max(all_nuc)
            - min-max(cls_nuc), normalized across the models of one domain.
  transrate TransRate (Huang et al., ICML 2022): R(Z) - (1/K) sum_k R(Z_k), R(Z) = 1/2 logdet(I + Z^T Z / (n eps)),
            eps = 1e-4, on centred L2-normalized embeddings.
Input: $FINETUNE_GEOM_EMB_DIR/<dataset>__<model>.npz. Output: results/robustness/transferability_more.csv
"""
import os
import sys
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy import linalg
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

from common import DOMAINS, RESULTS
from finding2_frozen_probes import EMB, MODELS


# ------------------------------------------------------------------ SFDA (official code, logging removed)
def _cov(X, shrinkage=-1):
    emp_cov = np.cov(np.asarray(X).T, bias=1)
    if shrinkage < 0:
        return emp_cov
    n_features = emp_cov.shape[0]
    mu = np.trace(emp_cov) / n_features
    shrunk_cov = (1.0 - shrinkage) * emp_cov
    shrunk_cov.flat[:: n_features + 1] += shrinkage * mu
    return shrunk_cov


def softmax(X):
    X = np.copy(X)
    X -= np.max(X, axis=1).reshape((-1, 1))
    np.exp(X, X)
    X /= np.sum(X, axis=1).reshape((-1, 1))
    return X


def _class_means(X, y):
    classes, y = np.unique(y, return_inverse=True)
    cnt = np.bincount(y)
    means = np.zeros(shape=(len(classes), X.shape[1]))
    np.add.at(means, y, X)
    means /= cnt[:, None]
    means_ = np.zeros(shape=(len(classes), X.shape[1]))
    for i in range(len(classes)):
        means_[i] = (np.sum(means, axis=0) - means[i]) / (len(classes) - 1)
    return means, means_


def iterative_A(A, max_iterations=3):
    x = A.sum(axis=1)
    for _ in range(max_iterations):
        temp = np.dot(A, x)
        y = temp / np.linalg.norm(temp, 2)
        temp = np.dot(A, y)
        x = temp / np.linalg.norm(temp, 2)
    return np.dot(np.dot(x.T, A), y)


class SFDA:
    def __init__(self, shrinkage=None):
        self.shrinkage = shrinkage

    def fit(self, X, y):
        classes, yt = np.unique(y, return_inverse=True)
        self.priors_ = np.bincount(yt) / float(len(y))
        cnt = np.bincount(yt)
        means = np.zeros(shape=(len(classes), X.shape[1]))
        np.add.at(means, yt, X)
        means /= cnt[:, None]
        Sw = np.zeros(shape=(X.shape[1], X.shape[1]))
        for idx, group in enumerate(classes):
            Sw += self.priors_[idx] * np.atleast_2d(_cov(X[y == group, :]))
        if self.shrinkage is None:
            self.shrinkage = max(np.exp(-5 * iterative_A(Sw, max_iterations=3)), 1e-10)
        St = _cov(X, shrinkage=self.shrinkage)
        n_features = Sw.shape[0]
        mu = np.trace(Sw) / n_features
        shrunk_Sw = (1.0 - self.shrinkage) * Sw
        shrunk_Sw.flat[:: n_features + 1] += self.shrinkage * mu
        evals, evecs = linalg.eigh(St - shrunk_Sw, shrunk_Sw)
        evecs = evecs[:, np.argsort(evals)[::-1]]
        self.coef_ = np.dot(means, evecs).dot(evecs.T)
        self.intercept_ = -0.5 * np.diag(np.dot(means, self.coef_.T)) + np.log(self.priors_)
        return self

    def predict_proba(self, X):
        return softmax(np.dot(X, self.coef_.T) + self.intercept_)


def sfda_score(X, y):
    X = X.copy()
    n = len(y)
    first = SFDA()
    prob = first.fit(X, y).predict_proba(X)
    prob = np.exp(prob) / np.exp(prob).sum(axis=1, keepdims=True)
    _, means_ = _class_means(X, y)
    for c in range(len(np.unique(y))):
        idx = np.where(y == c)[0]
        p = prob[idx, c].reshape(-1, 1)
        X[idx] = p * X[idx] + (1 - p) * means_[c]
    prob = SFDA(shrinkage=first.shrinkage).fit(X, y).predict_proba(X)
    return float(np.sum(prob[np.arange(n), y]) / n)


# ------------------------------------------------------------------ NCTI terms (official code, logging removed)
def ncti_terms(X, y):
    C = np.unique(y).shape[0]
    pca = PCA(n_components=64, random_state=0)
    X = pca.fit_transform(X)
    temp = max(np.exp(-pca.explained_variance_[:32].sum()), 1e-10)
    clf = LinearDiscriminantAnalysis(solver="svd") if temp == 1e-10 else LinearDiscriminantAnalysis(solver="eigen", shrinkage=float(temp))
    low = clf.fit_transform(X, y)
    low = low - low.mean(0, keepdims=True)
    all_nuc = np.linalg.norm(low, ord="nuc")
    pred = clf.predict_proba(X)
    ncc = np.sum(pred[np.arange(X.shape[0]), y]) / X.shape[0]
    cls_nuc = sum(np.linalg.norm(pred[y == c], ord="nuc") for c in range(C))
    return float(all_nuc), float(ncc), float(np.log(cls_nuc))


# ------------------------------------------------------------------ TransRate
def coding_rate(Z, eps):
    n, d = Z.shape
    return 0.5 * np.linalg.slogdet(np.eye(d) + Z.T @ Z / (n * eps))[1]


def transrate(X, y, eps=1e-4):
    Z = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    Z = Z - Z.mean(0, keepdims=True)
    K = len(np.unique(y))
    return float(coding_rate(Z, eps) - sum(coding_rate(Z[y == c], eps) for c in np.unique(y)) / K)


def job(args):
    dom, model = args
    p = EMB / f"{dom}__{model}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    X = z["embeddings"].astype(np.float64)
    _, y = np.unique(z["labels"].astype(int), return_inverse=True)
    a, b, c = ncti_terms(X, y)
    r = dict(dataset=dom, model=model, sfda=sfda_score(X, y), ncti_all_nuc=a, ncti_ncc=b, ncti_cls_nuc=c, transrate=transrate(X, y))
    print(r, flush=True)
    return r


def mm(s):
    return (s - s.min()) / (s.max() - s.min())


if __name__ == "__main__":
    with Pool(int(os.environ.get("NPROC", "8"))) as pool:
        rows = [r for r in pool.imap_unordered(job, [(d, m) for d in DOMAINS for m in MODELS]) if r]
    t = pd.DataFrame(rows).sort_values(["dataset", "model"])
    g = t.groupby("dataset")
    t["ncti"] = g.ncti_ncc.transform(mm) + g.ncti_all_nuc.transform(mm) - g.ncti_cls_nuc.transform(mm)
    t.to_csv(RESULTS / "robustness" / "transferability_more.csv", index=False)
    print(t.to_string(index=False))
