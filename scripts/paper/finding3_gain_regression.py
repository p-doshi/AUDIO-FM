"""Finding 3: does frozen geometry predict adaptation gain?

gain ~ difficulty + uniformity + domain_match on per-(model, domain, seed) rows (primary condition).
- OLS with model-clustered (CR1) standard errors, t with G-1 degrees of freedom.
- 95% CI for R^2 from 2,000 bootstrap resamples of models (default_rng(0)).
- Within-domain Spearman between uniformity and seed-averaged gain over (model, domain) pairs, with a
  model-bootstrap CI.
- Sensitivity: without machine (MIMII); speech LoRA restricted to complete seeds.
Output: OUT/finding3_regression.csv, OUT/finding3_rows.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import OUT, add_difficulty, gain_rows

PRED = ["difficulty", "uniformity", "domain_match"]


def ols(d, groups=True):
    y = d.gain.values.astype(float)
    A = np.column_stack([np.ones(len(d))] + [d[c].values.astype(float) for c in PRED])
    b, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = y - A @ b
    r2 = 1 - e @ e / np.sum((y - y.mean()) ** 2)
    out = dict(R2=r2, **{f"b_{c}": v for c, v in zip(PRED, b[1:])})
    if groups:
        g = d.model.values
        G = np.unique(g)
        n, k = A.shape
        bread = np.linalg.pinv(A.T @ A)
        meat = sum(np.outer(A[g == x].T @ e[g == x], A[g == x].T @ e[g == x]) for x in G)
        V = bread @ meat @ bread * (len(G) / (len(G) - 1)) * ((n - 1) / (n - k))
        se = np.sqrt(np.diag(V))
        p = 2 * stats.t.sf(np.abs(b / se), len(G) - 1)
        out.update({f"se_{c}": v for c, v in zip(PRED, se[1:])}, **{f"p_{c}": v for c, v in zip(PRED, p[1:])})
    return out


def within_rho(t):
    x = t.uniformity - t.groupby("dataset").uniformity.transform("mean")
    y = t.gain - t.groupby("dataset").gain.transform("mean")
    return stats.spearmanr(x, y)[0]


def main():
    df = gain_rows()
    df.to_csv(OUT / "finding3_rows.csv", index=False)
    rows = []

    base = ols(df)
    t = df.groupby(["dataset", "model"], as_index=False).agg(uniformity=("uniformity", "first"), gain=("gain", "mean"))
    rng = np.random.default_rng(0)
    models = df.model.unique()
    br2, brho = [], []
    for _ in range(2000):
        pick = rng.choice(models, len(models), replace=True)
        d = pd.concat([df[df.model == m] for m in pick])
        br2.append(ols(d, groups=False)["R2"])
        brho.append(within_rho(pd.concat([t[t.model == m] for m in pick])))
    rows.append(dict(variant="all public domains", n_rows=len(df), n_models=df.model.nunique(), **base,
                     R2_ci_lo=np.percentile(br2, 2.5), R2_ci_hi=np.percentile(br2, 97.5),
                     within_rho=within_rho(t), within_rho_ci_lo=np.percentile(brho, 2.5),
                     within_rho_ci_hi=np.percentile(brho, 97.5), n_pairs=len(t)))

    no_machine = gain_rows(exclude_domains=("mimii",))
    rows.append(dict(variant="without machine", n_rows=len(no_machine), n_models=no_machine.model.nunique(), **ols(no_machine)))

    complete = add_difficulty(df[(df.dataset != "librispeech") | df.seed.isin([0, 3])])
    rows.append(dict(variant="speech complete seeds only", n_rows=len(complete), n_models=complete.model.nunique(), **ols(complete)))

    cnt = df.groupby(["dataset", "model"]).seed.nunique()
    full = cnt[cnt == 5].index
    all5 = add_difficulty(df.set_index(["dataset", "model"]).loc[lambda x: x.index.isin(full)].reset_index())
    rows.append(dict(variant="(model, domain) pairs with all 5 seeds", n_rows=len(all5), n_models=all5.model.nunique(), **ols(all5)))

    out = pd.DataFrame(rows)
    out.to_csv(OUT / "finding3_regression.csv", index=False)
    print(out.T.to_string())


if __name__ == "__main__":
    main()
