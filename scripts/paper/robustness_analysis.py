"""Robustness analyses on the outputs of robustness_frozen.py (files in results/robustness/).

A. Uniformity -> frozen kNN accuracy and kNN - linear gap, for every kNN setting (k in {1, 5, 10, 20, 50};
   Euclidean on standardized features or cosine on raw embeddings), with accuracy or balanced accuracy, and for
   uniformity at t in {0.5, 1, 2, 5, 10}. Statistic as in Finding 2: Spearman after demeaning within domain.
   p: two-sided model-identity permutation (same permutation in every domain), 10,000 draws.
B. Shrinkage H-score and GBC vs adaptation gain, exactly as LogME in finding3_transferability.py.
C. Cross-domain stability (Finding 1 test, 10,000 permutations, BH over the 10 public pairs) for debiased linear
   CKA, linear CKA and mutual k-NN overlap, next to RSA.
D. Exploratory: the same test after removing each model's main effect (double-centring every matrix with the
   diagonal excluded), and the cross-domain Spearman of per-model centrality (mean similarity to the other models).
Outputs: OUT/robust_uniformity_knn.csv, OUT/robust_transferability.csv, OUT/robust_cross_domain.csv,
         OUT/robust_cross_domain_model_effect_removed.csv, OUT/robust_centrality.csv
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import DOMAIN_NAME, DOMAINS, OUT, RESULTS, bh, frozen_geometry, gain_rows, read_square
from finding1 import xdomain_p

R = RESULTS / "robustness"
N_PERM = 10000


def demean(df, c):
    return df[c] - df.groupby("dataset")[c].transform("mean")


def within_rho(df, x, y):
    return stats.spearmanr(demean(df, x), demean(df, y))[0]


def perm_p(df, x, y, rng, n=N_PERM):
    obs = within_rho(df, x, y)
    ms = df.model.unique()
    lut = {(a, b): v for a, b, v in zip(df.dataset, df.model, df[x])}
    c = 0
    for _ in range(n):
        mp = dict(zip(ms, rng.permutation(ms)))
        d = df.copy()
        d[x] = [lut.get((a, mp[b]), np.nan) for a, b in zip(df.dataset, df.model)]
        d = d.dropna(subset=[x])
        c += abs(within_rho(d, x, y)) >= abs(obs)
    return obs, (c + 1) / (n + 1)


def part_a():
    pr = pd.read_csv(R / "probes_robust.csv")
    geo = pd.read_csv(R / "geo_te_robust.csv")
    lin = pr[pr.probe == "linear_conv"].set_index(["dataset", "model"])
    rows = []
    rng = np.random.default_rng(0)
    for (metric, k), g in pr[pr.probe == "knn"].groupby(["metric", "k"]):
        d = g.merge(geo, on=["dataset", "model"])
        for score in ["acc", "bal_acc"]:
            d["knn"] = d[score]
            d["gap"] = d[score].values - lin.loc[list(zip(d.dataset, d.model)), score].values
            for target in ["knn", "gap"]:
                rho, p = perm_p(d, "uniformity_t2", target, rng, 2000)
                rows.append(dict(vary="kNN", metric=metric, k=int(k), score=score, t=2, target=target, n=len(d),
                                 rho=rho, p_model_perm=p, p_scipy=stats.spearmanr(demean(d, "uniformity_t2"), demean(d, target))[1]))
    base = pr[(pr.probe == "knn") & (pr.metric == "euclidean_std") & (pr.k == 10)].merge(geo, on=["dataset", "model"])
    base["knn"] = base.acc
    base["gap"] = base.acc.values - lin.loc[list(zip(base.dataset, base.model)), "acc"].values
    for t in [0.5, 1, 2, 5, 10]:
        for target in ["knn", "gap"]:
            rho, p = perm_p(base, f"uniformity_t{t:g}", target, rng, 2000)
            rows.append(dict(vary="t", metric="euclidean_std", k=10, score="acc", t=t, target=target, n=len(base), rho=rho,
                             p_model_perm=p, p_scipy=stats.spearmanr(demean(base, f"uniformity_t{t:g}"), demean(base, target))[1]))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "robust_uniformity_knn.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def part_b():
    rng = np.random.default_rng(0)
    est = pd.read_csv(R / "geo_te_robust.csv")[["dataset", "model", "h_alpha", "gbc"]]
    g = gain_rows().groupby(["dataset", "model"], as_index=False).gain.mean()
    m = est.merge(g, on=["dataset", "model"])
    ms = m.model.unique()
    rows = []
    for x in ["h_alpha", "gbc"]:
        obs, p = perm_p(m, x, "gain", rng, 5000)
        boot = []
        for _ in range(2000):
            pick = rng.choice(ms, len(ms))
            boot.append(within_rho(pd.concat([m[m.model == q] for q in pick]), x, "gain"))
        rows.append(dict(estimator=x, n_pairs=len(m), within_domain_rho=obs, perm_p=p,
                         ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5)))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "robust_transferability.csv", index=False)


def part_c():
    rows = []
    for metric in ["rsa", "dcka", "cka", "mknn"]:
        if metric == "rsa":
            D = {DOMAIN_NAME[d]: read_square(RESULTS / "geometry" / "rsa_frozen" / f"rsa_{d}.csv") for d in DOMAINS}
        else:
            D = {DOMAIN_NAME[d]: read_square(R / f"sim_{metric}_{d}.csv") for d in DOMAINS}
        common = sorted(set.intersection(*[set(M.index) for M in D.values()]))
        A = {k: M.loc[common, common].values.astype(float) for k, M in D.items()}
        part = []
        for a, b in itertools.combinations(A, 2):
            rho, p = xdomain_p(A[a], A[b], N_PERM, np.random.default_rng(0))
            part.append(dict(metric=metric, domain_a=a, domain_b=b, n_models=len(common), rho=rho, p_perm=p))
        part = pd.DataFrame(part)
        part["q_bh"] = bh(part.p_perm)
        rows.append(part)
    t = pd.concat(rows)
    t.to_csv(OUT / "robust_cross_domain.csv", index=False)
    print(t.pivot_table(index=["domain_a", "domain_b"], columns="metric", values=["rho", "q_bh"]).round(3).to_string())


def double_centre(M):
    A = M.copy()
    np.fill_diagonal(A, np.nan)
    r = np.nanmean(A, 1)
    return M - r[:, None] - r[None, :] + np.nanmean(A)


def load_metric(metric):
    if metric == "rsa":
        D = {DOMAIN_NAME[d]: read_square(RESULTS / "geometry" / "rsa_frozen" / f"rsa_{d}.csv") for d in DOMAINS}
    else:
        D = {DOMAIN_NAME[d]: read_square(R / f"sim_{metric}_{d}.csv") for d in DOMAINS}
    common = sorted(set.intersection(*[set(M.index) for M in D.values()]))
    return {k: M.loc[common, common].values.astype(float) for k, M in D.items()}, common


def part_d():
    rows, cent_rows = [], []
    for metric in ["rsa", "dcka", "mknn"]:
        A, common = load_metric(metric)
        part = []
        for a, b in itertools.combinations(A, 2):
            rho, p = xdomain_p(double_centre(A[a]), double_centre(A[b]), N_PERM, np.random.default_rng(0))
            part.append(dict(metric=metric, domain_a=a, domain_b=b, n_models=len(common), rho=rho, p_perm=p))
        part = pd.DataFrame(part)
        part["q_bh"] = bh(part.p_perm)
        rows.append(part)
        cent = pd.DataFrame({k: [np.mean(np.delete(M[i], i)) for i in range(len(common))] for k, M in A.items()}, index=common)
        c = cent.corr("spearman").stack().rename("rho_centrality").reset_index()
        c.insert(0, "metric", metric)
        cent_rows.append(c[c.level_0 < c.level_1])
    t = pd.concat(rows)
    t.to_csv(OUT / "robust_cross_domain_model_effect_removed.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    pd.concat(cent_rows).rename(columns={"level_0": "domain_a", "level_1": "domain_b"}).to_csv(OUT / "robust_centrality.csv", index=False)


if __name__ == "__main__":
    import sys
    for part in (sys.argv[1:] or ["a", "b", "c", "d"]):
        {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[part]()
