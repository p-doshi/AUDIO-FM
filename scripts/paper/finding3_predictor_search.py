"""Finding 3: exhaustive predictor search for adaptation gain (and adapted accuracy).

Unit: (model, domain), seed-averaged, five public domains; every variable demeaned within domain.
Score: leave-one-model-out cross-validated R^2 (all of a model's domains held out together), over every
subset of 1-3 predictors from a pool. Significance of the best subset: N_PERM permutations of model
identity (the same permutation in every domain), re-running the full search each time, so the null
includes the subset selection. default_rng(0).

Pools:
  label-free      uniformity, alignment, twonn, log_params, domain_match, discriminative
  + label-using   + knn, linear (frozen probes), frozen_acc (converged frozen probe)
  + post-hoc      + d_uni (change in uniformity during adaptation)
Targets: gain, adapted_acc. 2 targets x 3 pools = 6 searches (Bonferroni factor for the label-using result).
Output: OUT/finding3_predictor_search.csv
"""
import itertools
import os

import numpy as np
import pandas as pd

from common import OUT, RESULTS, frozen_geometry, gain_rows, primary_condition

N_PERM = int(os.environ.get("N_PERM", 2000))


def table():
    rows = gain_rows()
    rows["adapted_acc"] = rows.frozen_acc + rows.gain
    t = rows.groupby(["dataset", "model"], as_index=False).agg(
        gain=("gain", "mean"), adapted_acc=("adapted_acc", "mean"), frozen_acc=("frozen_acc", "mean"),
        domain_match=("domain_match", "first"))
    geo = frozen_geometry()[["dataset", "model", "uniformity", "alignment", "twonn"]]
    pr = pd.read_csv(RESULTS / "probes" / "frozen_knn_linear.csv")[["dataset", "model", "knn", "linear"]]
    par = pd.read_csv(RESULTS / "models" / "model_parameter_counts.csv", usecols=[0, 1])
    tax = pd.read_csv(RESULTS / "models" / "model_arch_taxonomy.csv")[["model", "type"]]
    au = pd.concat([pd.read_csv(RESULTS / "geometry" / "adaptation" / f"uniformity_{d}.csv")
                    for d in ["birdclef", "fma_genre", "librispeech", "mimii", "urbansound8k"]])
    au = au[~((au.dataset == "urbansound8k") & (au.condition == "lora") & (au.model == "wavlm"))]
    au = au[au.condition == au.model.map(primary_condition)].groupby(["dataset", "model"], as_index=False).d_uni.mean()
    t = (t.merge(geo, on=["dataset", "model"]).merge(pr, on=["dataset", "model"], how="left").merge(par, on="model")
         .merge(tax, on="model").merge(au, on=["dataset", "model"], how="left"))
    t["log_params"] = np.log10(t.n_params)
    t["discriminative"] = t.type.isin(["Supervised/discriminative", "Cross-modal contrastive"]).astype(float)
    return t


def loo_r2(y, X, groups):
    pred = np.empty_like(y)
    for g in np.unique(groups):
        te = groups == g
        tr = ~te
        A = np.column_stack([np.ones(tr.sum()), X[tr]])
        b, *_ = np.linalg.lstsq(A, y[tr], rcond=None)
        pred[te] = np.column_stack([np.ones(te.sum()), X[te]]) @ b
    return 1 - np.sum((y - pred) ** 2) / np.sum((y - y.mean()) ** 2)


def search(d, target, pool):
    y = d[target].values
    g = d.model.values
    res = [(loo_r2(y, d[list(c)].values, g), " + ".join(c)) for k in (1, 2, 3) for c in itertools.combinations(pool, k)]
    return sorted(res, reverse=True)


def main():
    LF = ["uniformity", "alignment", "twonn", "log_params", "domain_match", "discriminative"]
    LU = LF + ["knn", "linear", "frozen_acc"]
    PH = LU + ["d_uni"]
    t = table().dropna(subset=PH)
    print(f"n={len(t)} (model, domain) pairs, {t.model.nunique()} models")
    cols = PH + ["gain", "adapted_acc"]
    dm = t.copy()
    for c in cols:
        dm[c] = dm[c] - dm.groupby("dataset")[c].transform("mean")
    rng = np.random.default_rng(0)
    models = dm.model.unique()
    lut = dm.set_index(["dataset", "model"])
    out = []
    for target in ["gain", "adapted_acc"]:
        for label, pool in [("frozen, label-free", LF), ("frozen, incl. label-using", LU), ("incl. post-hoc d_uni", PH)]:
            r = search(dm, target, pool)
            uni = loo_r2(dm[target].values, dm[["uniformity"]].values, dm.model.values)
            null = []
            for _ in range(N_PERM):
                perm = dict(zip(models, rng.permutation(models)))
                d2 = dm.copy()
                d2[target] = [lut[target].get((a, perm[m]), np.nan) for a, m in zip(dm.dataset, dm.model)]
                d2 = d2.dropna(subset=[target])
                null.append(search(d2, target, pool)[0][0])
            p = (1 + np.sum(np.array(null) >= r[0][0])) / (1 + len(null))
            out.append(dict(target=target, pool=label, n_pairs=len(dm), uniformity_alone_cv_r2=uni, best=r[0][1],
                            best_cv_r2=r[0][0], perm_p=p, p_bonferroni_6=min(1.0, 6 * p),
                            second=r[1][1], second_cv_r2=r[1][0], third=r[2][1], third_cv_r2=r[2][0]))
            print(out[-1], flush=True)
    pd.DataFrame(out).to_csv(OUT / "finding3_predictor_search.csv", index=False)


if __name__ == "__main__":
    main()
