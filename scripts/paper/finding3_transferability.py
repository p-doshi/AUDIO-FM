"""Finding 3: label-based transferability estimators vs adaptation gain.

Estimators (computed on frozen embeddings, five public domains; results/probes/transferability_estimators.csv):
  logme             LogME (You et al., 2021) on frozen embeddings and labels
  knn_ref_accuracy  kNN accuracy with up to 100 labelled reference clips per class
  uniformity        for reference
Target: seed-averaged converged-probe gain (primary condition), per (model, domain).
Statistic: within-domain Spearman (both variables demeaned within domain).
p: two-sided permutation of model identity (same permutation in every domain), 5,000 draws, default_rng(0).
CI: 2,000 bootstrap resamples of models.
Output: OUT/finding3_transferability.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import OUT, RESULTS, gain_rows


def within(df, x, y):
    a = df[x] - df.groupby("dataset")[x].transform("mean")
    b = df[y] - df.groupby("dataset")[y].transform("mean")
    return stats.spearmanr(a, b)[0]


def main():
    rng = np.random.default_rng(0)
    est = pd.read_csv(RESULTS / "probes" / "transferability_estimators.csv").rename(columns={"domain": "dataset"})
    est = est.drop(columns=[c for c in ["gain"] if c in est.columns])
    g = gain_rows().groupby(["dataset", "model"], as_index=False).gain.mean()
    m = est.merge(g, on=["dataset", "model"])
    ms = m.model.unique()
    rows = []
    for x in ["logme", "knn_ref_accuracy", "uniformity"]:
        obs = within(m, x, "gain")
        lut = {(a, b): v for a, b, v in zip(m.dataset, m.model, m[x])}
        c = 0
        for _ in range(5000):
            mp = dict(zip(ms, rng.permutation(ms)))
            d = m.copy()
            d[x] = [lut.get((a, mp[b]), np.nan) for a, b in zip(m.dataset, m.model)]
            c += abs(within(d.dropna(subset=[x]), x, "gain")) >= abs(obs)
        boot = []
        for _ in range(2000):
            pick = rng.choice(ms, len(ms))
            boot.append(within(pd.concat([m[m.model == q] for q in pick]), x, "gain"))
        rows.append(dict(estimator=x, n_pairs=len(m), within_domain_rho=obs, perm_p=(c + 1) / 5001,
                         ci_lo=np.percentile(boot, 2.5), ci_hi=np.percentile(boot, 97.5)))
        print(rows[-1])
    pd.DataFrame(rows).to_csv(OUT / "finding3_transferability.csv", index=False)


if __name__ == "__main__":
    main()
