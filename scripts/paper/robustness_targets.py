"""Further robustness checks for Findings 1-3 (aggregate inputs only; five public domains).

A. Gain targets. Every frozen estimator (uniformity, LogME, kNN with labelled reference clips, shrinkage H-score, GBC,
   SFDA, NCTI, TransRate; the last three from transferability_more.py)
   against three per-(model, domain) targets, seeds averaged: raw gain (adapted - frozen accuracy, the paper's
   target), normalized gain (adapted - frozen) / (1 - frozen), and adapted accuracy itself (what transferability
   estimators are designed to rank). Within-domain Spearman, two-sided model-identity permutation p (5,000 draws),
   95% CI from 2,000 model bootstraps.  -> OUT/robust_gain_targets.csv
B. Regression variants. gain ~ difficulty + uniformity + domain match (model-clustered SE, t with G - 1 df) with
   (i) difficulty computed leaving the row's own model out, and (ii) normalized gain as the target.
   -> OUT/robust_regression_variants.csv
C. Pretraining subsets. Finding 1 test (one-sided model-identity permutation, 10,000 draws, BH over the 10 public
   pairs) restricted to the 10 encoders not pretrained on speech and to the 9 speech-pretrained encoders, for RSA and
   linear CKA.  -> OUT/robust_cross_domain_subsets.csv
D. Discriminative vs reconstruction kNN/linear ratio at model level: ratio demeaned within domain, difference of group
   means over all (model, domain) points, two-sided permutation of model labels (same in every domain, 10,000 draws),
   for both definitions of "discriminative" used in finding2.py.  -> OUT/robust_ratio_groups.csv
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import (CHANCE, DOMAIN_NAME, DOMAINS, OUT, RESULTS, SPEECH_MODELS, bh, difficulty, gain_rows,
                    load_converged_probe, primary_condition, read_square)
from finding1 import xdomain_p
from finding3_gain_regression import ols

R = RESULTS / "robustness"


def demean(df, c):
    return df[c] - df.groupby("dataset")[c].transform("mean")


def within(df, x, y):
    return stats.spearmanr(demean(df, x), demean(df, y))[0]


def _wrho(X, Y):
    """Within-domain Spearman on (domain x model) grids: demean each row, then Spearman over all cells."""
    x = (X - X.mean(1, keepdims=True)).ravel()
    y = (Y - Y.mean(1, keepdims=True)).ravel()
    return np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1]


ESTIMATORS = ["uniformity", "logme", "knn_ref_accuracy", "h_alpha", "gbc", "sfda", "ncti", "transrate"]


def part_a():
    est = pd.read_csv(RESULTS / "probes" / "transferability_estimators.csv").rename(columns={"domain": "dataset"})
    est = est[["dataset", "model", "logme", "knn_ref_accuracy", "uniformity"]]
    est = est.merge(pd.read_csv(R / "geo_te_robust.csv")[["dataset", "model", "h_alpha", "gbc"]], on=["dataset", "model"])
    est = est.merge(pd.read_csv(R / "transferability_more.csv")[["dataset", "model", "sfda", "ncti", "transrate"]], on=["dataset", "model"])
    g = gain_rows()
    g["gain_norm"] = g.gain / (1 - g.frozen_acc)
    t = g.groupby(["dataset", "model"], as_index=False)[["gain", "gain_norm", "adapted_acc", "frozen_acc"]].mean()
    m = est.merge(t, on=["dataset", "model"])
    grid = {c: m.pivot(index="dataset", columns="model", values=c) for c in ESTIMATORS + ["gain", "gain_norm", "adapted_acc", "frozen_acc"]}
    assert not any(v.isna().any().any() for v in grid.values()), "every (domain, model) cell must be present"
    grid = {c: v.values.astype(float) for c, v in grid.items()}
    nm = grid["gain"].shape[1]
    rng = np.random.default_rng(0)
    rows = []
    for target in ["gain", "gain_norm", "adapted_acc", "frozen_acc"]:
        Y = grid[target]
        for x in ESTIMATORS:
            X = grid[x]
            obs = _wrho(X, Y)
            null = np.array([_wrho(X[:, rng.permutation(nm)], Y) for _ in range(5000)])
            boot = []
            for _ in range(2000):
                pick = rng.choice(nm, nm)
                boot.append(_wrho(X[:, pick], Y[:, pick]))
            rows.append(dict(target=target, estimator=x, n_pairs=Y.size, within_domain_rho=obs,
                             perm_p=(1 + np.sum(np.abs(null) >= abs(obs))) / 5001,
                             ci_lo=np.nanpercentile(boot, 2.5), ci_hi=np.nanpercentile(boot, 97.5)))
            print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in rows[-1].items()}, flush=True)
    pd.DataFrame(rows).to_csv(OUT / "robust_gain_targets.csv", index=False)


def part_b():
    df = gain_rows()
    rows = [dict(variant="paper (difficulty from all models)", n_rows=len(df), **ols(df))]
    d = df.copy()
    pair = df.groupby(["dataset", "model"]).frozen_acc.mean()
    lomo = {}
    for (ds, mo) in pair.index:
        others = pair.loc[ds].drop(mo)
        lomo[(ds, mo)] = difficulty(others.mean(), CHANCE[ds])
    d["difficulty"] = [lomo[(a, b)] for a, b in zip(d.dataset, d.model)]
    rows.append(dict(variant="difficulty leaving own model out", n_rows=len(d), **ols(d)))
    n = df.copy()
    n["gain"] = n.gain / (1 - n.frozen_acc)
    rows.append(dict(variant="normalized gain (adapted - frozen) / (1 - frozen)", n_rows=len(n), **ols(n)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "robust_regression_variants.csv", index=False)
    print(out.T.to_string())


def part_c():
    rows = []
    for metric in ["rsa", "cka"]:
        D = {DOMAIN_NAME[d]: read_square(RESULTS / "geometry" / "rsa_frozen" / f"{metric}_{d}.csv") for d in DOMAINS}
        common = sorted(set.intersection(*[set(M.index) for M in D.values()]))
        for subset, keep in [("not speech-pretrained", [m for m in common if m not in SPEECH_MODELS]),
                             ("speech-pretrained", [m for m in common if m in SPEECH_MODELS])]:
            A = {k: M.loc[keep, keep].values.astype(float) for k, M in D.items()}
            part = []
            for a, b in itertools.combinations(A, 2):
                rho, p = xdomain_p(A[a], A[b], 10000, np.random.default_rng(0))
                part.append(dict(metric=metric, subset=subset, domain_a=a, domain_b=b, n_models=len(keep), rho=rho, p_perm=p))
            part = pd.DataFrame(part)
            part["q_bh"] = bh(part.p_perm)
            rows.append(part)
    t = pd.concat(rows)
    t.to_csv(OUT / "robust_cross_domain_subsets.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def part_d():
    pr = pd.read_csv(RESULTS / "probes" / "frozen_knn_linear.csv")
    tax = pd.read_csv(RESULTS / "models" / "model_arch_taxonomy.csv")[["model", "type"]]
    d = pr.merge(tax, on="model")
    d["ratio"] = d.knn / d.linear
    d["ratio_dm"] = demean(d, "ratio")
    rec = set(d[d.type.str.startswith("Masked")].model)
    defs = {"AST and CLAP": {"ast", "clap"},
            "all supervised or contrastive": set(d[d.type.isin(["Supervised/discriminative", "Cross-modal contrastive"])].model)}
    rng = np.random.default_rng(0)
    rows = []
    for name, disc in defs.items():
        q = d[d.model.isin(disc | rec)]
        ms = q.model.unique()

        def stat(lab):
            g = q.model.map(lab)
            return q.ratio_dm[g].mean() - q.ratio_dm[~g].mean()

        lab = {m: m in disc for m in ms}
        obs = stat(lab)
        flags = np.array([lab[m] for m in ms])
        null = [stat(dict(zip(ms, rng.permutation(flags)))) for _ in range(10000)]
        p = (1 + np.sum(np.abs(null) >= abs(obs))) / 10001
        rows.append(dict(discriminative=name, n_disc=len(disc), n_reconstruction=len(rec), n_points=len(q),
                         mean_diff_demeaned_ratio=obs, p_model_perm=p))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "robust_ratio_groups.csv", index=False)


if __name__ == "__main__":
    import sys
    for part in (sys.argv[1:] or ["b", "c", "d", "a"]):
        {"a": part_a, "b": part_b, "c": part_c, "d": part_d}[part]()
