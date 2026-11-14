"""Finding 3 (second part): how adaptation reshapes geometry, and whether that explains gain.

1. Shared offset: ||mean of L2-normalized embeddings||^2, frozen vs adapted, per valid adaptation run
   (from the stored Gram matrices; results/geometry/adaptation/spectral_*.csv, column mean_energy).
2. Delta-RSA per domain and method: mean over models of (mean adapted RSA to the other models - mean frozen
   RSA to the same models), per seed; reported as mean and SD over seeds. WavLM excluded in every domain.
3. Preserved ordering: Spearman across models between frozen and seed-averaged adapted mean RSA, per
   domain x method (10 cells), permutation p (50,000 draws, default_rng(0)), BH over the 10 cells.
4. Self-RSA links at model level (seeds averaged), per domain x condition:
   a. uniformity -> self-RSA (Spearman)
   b. self-RSA -> gain, partial Spearman controlling for uniformity (rank residualization)
   gain = converged-probe gain of that condition. Permutation p: exact over all orderings when n <= 8,
   otherwise 20,000 draws (default_rng(0)). BH over the cells.
Outputs: OUT/finding3_shared_offset.csv, OUT/finding3_delta_rsa.csv, OUT/finding3_self_rsa_links.csv
"""
import glob
import itertools
import math

import numpy as np
import pandas as pd
from scipy import stats

from common import DOMAINS, OUT, RESULTS, bh, load_converged_probe, read_square

G = RESULTS / "geometry"


def shared_offset():
    rows = []
    for f in sorted(glob.glob(str(G / "adaptation" / "spectral_*.csv"))):
        d = pd.read_csv(f)
        rows.append(d[["dataset", "model", "condition", "seed", "mean_energy_frozen", "mean_energy_adapted"]])
    d = pd.concat(rows)
    d["falls"] = d.mean_energy_adapted < d.mean_energy_frozen
    d.to_csv(OUT / "finding3_shared_offset.csv", index=False)
    print(f"shared offset falls in {d.falls.sum()} of {len(d)} valid adaptation runs")
    print("exceptions:\n", d[~d.falls][["dataset", "model", "condition", "seed"]].to_string(index=False))


def mean_rsa(M, models):
    return {x: M.loc[x, [o for o in models if o != x]].mean() for x in models}


def delta_rsa():
    rng = np.random.default_rng(0)
    rows = []
    for ds in DOMAINS:
        F = read_square(G / "rsa_frozen" / f"rsa_{ds}.csv")
        for cond in ["lora", "allora"]:
            per_d, per_order = [], []
            mats = sorted(glob.glob(str(G / "rsa_adapted" / f"rsa_{ds}_{cond}_seed[0-9].csv")))
            for f in mats:
                A = read_square(f)
                ms = [x for x in A.index if x in F.index and x != "wavlm"]
                fr, ad = mean_rsa(F, ms), mean_rsa(A, ms)
                per_d.append(np.mean([ad[x] - fr[x] for x in ms]))
                ms2 = [x for x in A.index if x in F.index]
                per_order.append(mean_rsa(A, ms2))
            models = sorted(set.intersection(*[set(p) for p in per_order]))
            ad = np.array([np.mean([p[x] for p in per_order]) for x in models])
            fr = np.array([F.loc[x, [o for o in models if o != x]].mean() for x in models])
            r = stats.spearmanr(fr, ad)[0]
            null = np.array([stats.spearmanr(fr, rng.permutation(ad))[0] for _ in range(50000)])
            p = (np.sum(np.abs(null) >= abs(r) - 1e-12) + 1) / 50001
            rows.append(dict(domain=ds, method=cond, seeds=len(mats), mean_delta_rsa=np.mean(per_d),
                             sd_over_seeds=np.std(per_d, ddof=1), n_models_order=len(models), rho_order=r, p_order=p))
    t = pd.DataFrame(rows)
    t["q_order"] = bh(t.p_order)
    t.to_csv(OUT / "finding3_delta_rsa.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


def partial(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    return np.corrcoef(ex, ey)[0, 1]


def perm_p(stat, x, y, rng, mc=20000):
    obs = stat(x, y)
    n = len(y)
    if not np.isfinite(obs):  # e.g. partial correlation undefined when the control fully determines one rank vector
        return obs, np.nan
    if n <= 8:
        perms = [np.array(p) for p in itertools.permutations(range(n))]
        c = sum(abs(stat(x, y[p])) >= abs(obs) - 1e-12 for p in perms)
        return obs, c / math.factorial(n)
    c = sum(abs(stat(x, y[rng.permutation(n)])) >= abs(obs) - 1e-12 for _ in range(mc))
    return obs, (c + 1) / (mc + 1)


def self_rsa_links():
    rng = np.random.default_rng(0)
    cells = pd.read_csv(G / "adaptation" / "self_rsa_model_level.csv")[["model", "self_rsa", "uniformity", "domain", "condition"]]
    g = pd.concat([load_converged_probe(), load_converged_probe(allora=True)])
    g = g.groupby(["dataset", "condition", "model"], as_index=False).gain.mean().rename(columns={"dataset": "domain"})
    cells = cells.merge(g, on=["domain", "condition", "model"], how="left")
    rows = []
    for (d, c), q in cells.groupby(["domain", "condition"]):
        q = q.dropna(subset=["self_rsa", "uniformity"])
        r1, p1 = perm_p(lambda a, b: stats.spearmanr(a, b)[0], q.uniformity.values, q.self_rsa.values, rng)
        qg = q.dropna(subset=["gain"])
        z = qg.uniformity.values
        r2, p2 = perm_p(lambda a, b: partial(a, b, z), qg.self_rsa.values, qg.gain.values, rng)
        rows.append(dict(domain=d, condition=c, n_models=len(q), rho_uniformity_selfrsa=r1, p_first=p1,
                         n_models_gain=len(qg), partial_rho_selfrsa_gain=r2, p_second=p2))
    t = pd.DataFrame(rows)
    t["q_first"] = bh(t.p_first)
    ok = t.p_second.notna()
    t.loc[ok, "q_second"] = bh(t.p_second[ok])
    t.to_csv(OUT / "finding3_self_rsa_links.csv", index=False)
    print(t.to_string(index=False, float_format=lambda v: f"{v:.4f}"))


if __name__ == "__main__":
    shared_offset()
    delta_rsa()
    self_rsa_links()
