"""Layer-wise analyses (outputs of layerwise_frozen.py in results/robustness/).

A. Uniformity -> kNN accuracy and -> kNN - linear gap at five relative depths (0, 0.25, 0.5, 0.75, 1; for every model
   the block nearest that depth). Statistic as in Finding 2: Spearman after demeaning within domain; p: two-sided
   model-identity permutation (same permutation in every domain), 5,000 draws.  -> OUT/layerwise_uniformity_knn.csv
   Also the relative depth of each model's best kNN and best linear block per domain.  -> OUT/layerwise_best_depth.csv
B. Cross-domain stability (Finding 1 test: one-sided model-identity permutation, 10,000 draws, BH over the 10 public
   pairs) of the mid-depth RSA and debiased-CKA matrices, next to the final-embedding RSA.  -> OUT/layerwise_cross_domain.csv
"""
import itertools

import numpy as np
import pandas as pd
from scipy import stats

from common import DOMAIN_NAME, DOMAINS, OUT, RESULTS, bh, read_square
from finding1 import xdomain_p

R = RESULTS / "robustness"
DEPTHS = [0.0, 0.25, 0.5, 0.75, 1.0]


def _wrho(X, Y):
    x = (X - X.mean(1, keepdims=True)).ravel()
    y = (Y - Y.mean(1, keepdims=True)).ravel()
    return np.corrcoef(stats.rankdata(x), stats.rankdata(y))[0, 1]


def part_a():
    d = pd.read_csv(R / "layerwise_probes.csv")
    d["gap"] = d.knn - d.linear
    rng = np.random.default_rng(0)
    rows = []
    for depth in DEPTHS:
        pick = d.loc[d.assign(dist=(d.depth - depth).abs()).groupby(["dataset", "model"]).dist.idxmin()]
        grid = {c: pick.pivot(index="dataset", columns="model", values=c) for c in ["uniformity", "knn", "gap", "linear"]}
        grid = {c: v.dropna(axis=1).values.astype(float) for c, v in grid.items()}
        nm = grid["uniformity"].shape[1]
        for target in ["knn", "gap"]:
            obs = _wrho(grid["uniformity"], grid[target])
            null = np.array([_wrho(grid["uniformity"][:, rng.permutation(nm)], grid[target]) for _ in range(5000)])
            rows.append(dict(depth=depth, target=target, n_pairs=grid["uniformity"].size, rho=obs,
                             p_model_perm=(1 + np.sum(np.abs(null) >= abs(obs))) / 5001,
                             mean_knn=grid["knn"].mean(), mean_linear=grid["linear"].mean()))
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "layerwise_uniformity_knn.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.3f}"))
    best = []
    for (ds, m), g in d.groupby(["dataset", "model"]):
        best.append(dict(dataset=ds, model=m, n_layers=g.n_layers.iloc[0],
                         best_knn_depth=g.loc[g.knn.idxmax(), "depth"], best_linear_depth=g.loc[g.linear.idxmax(), "depth"],
                         knn_best_minus_last=g.knn.max() - g.loc[g.depth.idxmax(), "knn"],
                         linear_best_minus_last=g.linear.max() - g.loc[g.depth.idxmax(), "linear"]))
    b = pd.DataFrame(best)
    b.to_csv(OUT / "layerwise_best_depth.csv", index=False)
    print(b.groupby("dataset")[["best_knn_depth", "best_linear_depth", "knn_best_minus_last", "linear_best_minus_last"]].mean().round(3))


def part_b():
    rows = []
    sets = {}
    for metric, src in [("final rsa", "rsa"), ("mid rsa", "layer_mid_rsa"), ("mid dcka", "layer_mid_dcka")]:
        if metric == "final rsa":
            sets[metric] = {DOMAIN_NAME[d]: read_square(RESULTS / "geometry" / "rsa_frozen" / f"rsa_{d}.csv") for d in DOMAINS}
        else:
            sets[metric] = {DOMAIN_NAME[d]: read_square(R / f"{src}_{d}.csv") for d in DOMAINS}
    # one model set for all three measures (EnCodecMAE has no layer-wise run; MIMII has no final-RSA Conformer row)
    common = sorted(set.intersection(*[set(M.index) for D in sets.values() for M in D.values()]))
    for metric, D in sets.items():
        A = {k: M.loc[common, common].values.astype(float) for k, M in D.items()}
        part = []
        for a, b in itertools.combinations(A, 2):
            rho, p = xdomain_p(A[a], A[b], 10000, np.random.default_rng(0))
            part.append(dict(metric=metric, domain_a=a, domain_b=b, n_models=len(common), rho=rho, p_perm=p))
        part = pd.DataFrame(part)
        part["q_bh"] = bh(part.p_perm)
        rows.append(part)
    t = pd.concat(rows)
    t.to_csv(OUT / "layerwise_cross_domain.csv", index=False)
    print(t.pivot_table(index=["domain_a", "domain_b"], columns="metric", values=["rho", "q_bh"]).round(3).to_string())


NATIVE_HEAD = ["ast", "clap", "musicfm", "panns_cnn14"]  # clip embedding is not the mean of the last block


def _perm_test(d, y, rng, n=5000):
    """Finding 2 statistic on a long table (dataset, model, uniformity, y): within-domain Spearman, model permutation."""
    def wr(u, t):
        return stats.spearmanr(u - u.groupby(d.dataset).transform("mean"), t - t.groupby(d.dataset).transform("mean"))[0]
    obs = wr(d.uniformity, d[y])
    ms = d.model.unique()
    lut = {(a, b): v for a, b, v in zip(d.dataset, d.model, d.uniformity)}
    c = 0
    for _ in range(n):
        mp = dict(zip(ms, rng.permutation(ms)))
        u = pd.Series([lut.get((a, mp[b]), np.nan) for a, b in zip(d.dataset, d.model)], index=d.index)
        ok = u.notna()
        q = d[ok]
        c += abs(stats.spearmanr(u[ok] - u[ok].groupby(q.dataset).transform("mean"),
                                 q[y] - q.groupby("dataset")[y].transform("mean"))[0]) >= abs(obs)
    return obs, (c + 1) / (n + 1)


def part_c():
    """Finding 2 on the native clip embeddings versus the mean-pooled last block of every model."""
    from common import frozen_geometry
    nat = (pd.read_csv(RESULTS / "probes" / "frozen_knn_linear.csv")
           .merge(pd.read_csv(RESULTS / "probes" / "frozen_linear_iterations.csv")[["dataset", "model", "lin_conv"]], on=["dataset", "model"])
           .merge(frozen_geometry()[["dataset", "model", "uniformity"]], on=["dataset", "model"]))
    nat["gap"] = nat.knn - nat.lin_conv
    L = pd.read_csv(R / "layerwise_probes.csv")
    last = L.loc[L.groupby(["dataset", "model"]).depth.idxmax()].reset_index(drop=True)
    last["gap"] = last.knn - last.linear
    rng = np.random.default_rng(0)
    rows = []
    for label, d in [("native clip embedding, all models", nat),
                     ("native, without AST", nat[nat.model != "ast"]),
                     ("native, without AST, CLAP, MusicFM, PANNs", nat[~nat.model.isin(NATIVE_HEAD)]),
                     ("mean-pooled last block, all models", last),
                     ("mean-pooled last block, without AST, CLAP, MusicFM, PANNs", last[~last.model.isin(NATIVE_HEAD)])]:
        for y in ["knn", "gap"]:
            rho, p = _perm_test(d.reset_index(drop=True), y, rng)
            rows.append(dict(embedding=label, target=y, n_pairs=len(d), n_models=d.model.nunique(), rho=rho, p_model_perm=p))
            print(rows[-1], flush=True)
    loo = {}
    for m in sorted(nat.model.unique()):
        d = nat[nat.model != m]
        loo[m] = stats.spearmanr(d.uniformity - d.groupby("dataset").uniformity.transform("mean"),
                                 d.knn - d.groupby("dataset").knn.transform("mean"))[0]
    t = pd.DataFrame(rows)
    t.to_csv(OUT / "layerwise_native_vs_lastblock.csv", index=False)
    pd.Series(loo, name="rho_uniformity_knn_without_model").rename_axis("model").to_csv(OUT / "finding2_leave_one_model_out.csv")


if __name__ == "__main__":
    import sys
    for part in (sys.argv[1:] or ["a", "b", "c"]):
        {"a": part_a, "b": part_b, "c": part_c}[part]()
