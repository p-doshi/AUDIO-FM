"""Finding 1: cross-domain stability of model similarity, and taxonomy alignment.

Inputs: one 19 x 19 frozen RSA matrix per domain (results/geometry/rsa_frozen/rsa_<dataset>.csv).
The vessel matrix is not distributed; pass its path with ECHO_VESSEL_RSA to include it (as in the paper).

A. Cross-domain stability. For each pair of domains: Spearman between the two matrices' upper triangles
   (171 model pairs). One-sided permutation test that shuffles model identities (rows and columns together)
   in the first matrix, 10,000 permutations, default_rng(0) reset per pair. BH over all pairs.
   Robustness: the test is repeated with each model removed (5,000 permutations), BH within each roster.
B. Taxonomy alignment. Mantel r = Pearson(upper-triangle RSA, same-group indicator) for two groupings
   (audio_comp.pipelines.breadth_hypothesis_check: DOMAIN_PARADIGM_GROUPS, BREADTH_GROUPS). One-sided
   model-identity permutation test (N_PERM_MANTEL = 200,000, default_rng(0)), 95% CI from a bootstrap
   over models (10,000 resamples, default_rng(1); duplicate draws of one model never form a pair).
   BH over the domain x grouping tests. Leave-one-model-out: 10,000 permutations per roster.
Outputs: OUT/finding1_cross_domain.csv, OUT/finding1_cross_domain_loo.csv, OUT/finding1_taxonomy.csv,
         OUT/finding1_taxonomy_loo.csv
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd
from scipy import stats

from common import DOMAIN_NAME, DOMAINS, OUT, REPO, RESULTS, bh, read_square

sys.path.insert(0, str(REPO))
from audio_comp.pipelines.breadth_hypothesis_check import BREADTH_GROUPS, DOMAIN_PARADIGM_GROUPS  # noqa: E402

N_PERM_MANTEL = int(os.environ.get("N_PERM_MANTEL", 200000))
N_BOOT = int(os.environ.get("N_BOOT", 10000))


def matrices():
    D = {DOMAIN_NAME[d]: read_square(RESULTS / "geometry" / "rsa_frozen" / f"rsa_{d}.csv") for d in DOMAINS}
    if os.environ.get("ECHO_VESSEL_RSA"):
        D["vessel"] = read_square(os.environ["ECHO_VESSEL_RSA"])
    common = sorted(set.intersection(*[set(M.index) for M in D.values()]))
    return {k: M.loc[common, common].values.astype(float) for k, M in D.items()}, common


def xdomain_p(X, Y, n_perm, rng):
    n = len(X)
    iu = np.triu_indices(n, 1)
    yr = stats.rankdata(Y[iu])
    rho = np.corrcoef(stats.rankdata(X[iu]), yr)[0, 1]
    null = [np.corrcoef(stats.rankdata(X[np.ix_(p, p)][iu]), yr)[0, 1] for p in (rng.permutation(n) for _ in range(n_perm))]
    return rho, (1 + np.sum(np.array(null) >= rho)) / (n_perm + 1)


def cross_domain(A, models):
    rows = []
    for a, b in itertools.combinations(A, 2):
        rho, p = xdomain_p(A[a], A[b], 10000, np.random.default_rng(0))
        rows.append(dict(domain_a=a, domain_b=b, n_models=len(models), rho=rho, p_perm=p))
    t = pd.DataFrame(rows)
    t["q_bh"] = bh(t.p_perm)
    t.to_csv(OUT / "finding1_cross_domain.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    loo = []
    pairs = list(itertools.combinations(A, 2))
    for drop in [None] + models:
        keep = [i for i, m in enumerate(models) if m != drop]
        ps = [xdomain_p(A[a][np.ix_(keep, keep)], A[b][np.ix_(keep, keep)], 5000, np.random.default_rng(0))[1] for a, b in pairs]
        q = bh(ps)
        loo.append(dict(dropped=drop or "none", n_significant=int((q <= 0.05).sum()),
                        significant_pairs=";".join(f"{a}-{b}" for (a, b), v in zip(pairs, q) if v <= 0.05)))
    pd.DataFrame(loo).to_csv(OUT / "finding1_cross_domain_loo.csv", index=False)


def mantel_r(M, g, idx):
    a, b = np.triu_indices(len(idx), 1)
    i, j = idx[a], idx[b]
    keep = i != j
    x = M[i[keep], j[keep]]
    s = (g[i[keep]] == g[j[keep]]).astype(float)
    if s.std() == 0 or x.std() == 0:
        return np.nan
    return np.corrcoef(x, s)[0, 1]


def taxonomy(A, models):
    rows, loo = [], []
    for dom, M in A.items():
        for tax, groups in [("domain+paradigm", DOMAIN_PARADIGM_GROUPS), ("breadth", BREADTH_GROUPS)]:
            gmap = {m: k for k, ms in groups.items() for m in ms}
            g = np.array([gmap.get(m, f"_single_{m}") for m in models])
            n = len(models)
            base = np.arange(n)
            rng = np.random.default_rng(0)
            obs = mantel_r(M, g, base)
            null = np.array([mantel_r(M, g[rng.permutation(n)], base) for _ in range(N_PERM_MANTEL)])
            rb = np.random.default_rng(1)
            boot = np.array([mantel_r(M, g, rb.integers(0, n, n)) for _ in range(N_BOOT)])
            lo, hi = np.nanpercentile(boot, [2.5, 97.5])
            rows.append(dict(domain=dom, grouping=tax, n_models=n, same_group_pairs=int(sum(g[i] == g[j] for i, j in zip(*np.triu_indices(n, 1)))),
                             mantel_r=obs, p_perm=(1 + np.sum(null >= obs)) / (N_PERM_MANTEL + 1), ci_lo=lo, ci_hi=hi))
            for k, m in enumerate(models):
                keep = np.array([i for i in range(n) if i != k])
                Mk, gk = M[np.ix_(keep, keep)], g[keep]
                rk = np.random.default_rng(0)
                ok = mantel_r(Mk, gk, np.arange(n - 1))
                nk = np.array([mantel_r(Mk, gk[rk.permutation(n - 1)], np.arange(n - 1)) for _ in range(10000)])
                loo.append(dict(domain=dom, grouping=tax, dropped=m, mantel_r=ok, p_perm=(1 + np.sum(nk >= ok)) / 10001))
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    t["q_bh"] = bh(t.p_perm)
    t.to_csv(OUT / "finding1_taxonomy.csv", index=False)
    pd.DataFrame(loo).to_csv(OUT / "finding1_taxonomy_loo.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    A, models = matrices()
    print(f"{len(A)} domains, {len(models)} models")
    cross_domain(A, models)
    taxonomy(A, models)
