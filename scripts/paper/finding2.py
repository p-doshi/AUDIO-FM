"""Finding 2: what makes frozen embeddings usable.

Input: results/probes/frozen_knn_linear.csv (per dataset x model frozen linear and kNN test accuracy,
written by finding2_frozen_probes.py), results/geometry/frozen_geometry.csv (uniformity),
results/models/model_parameter_counts.csv and model_arch_taxonomy.csv.

Reports:
- per-domain ranks of discriminatively trained models on linear and kNN accuracy
- kNN/linear ratio, discriminative (supervised + contrastive) vs reconstruction (masked modelling) models,
  per domain, with a two-sided sign test over the five public domains
- Bird-MAE / AudioMAE accuracy ratios (same architecture and objective, different training domain)
- pooled Spearman, after demeaning every variable within domain, of uniformity with kNN accuracy and
  with the kNN - linear gap (linear = the same logistic regression run to convergence, max_iter 10,000,
  tol 1e-6; results/probes/frozen_linear_iterations.csv, column lin_conv), and the partial correlation
  controlling for log10 parameter count (rank residualization), with the VIF of the two predictors
The sign test is reported for two definitions of "discriminative": AST and CLAP (the two models with a
direct classification or contrastive objective on general audio), and all supervised or contrastive
models (AST, CLAP, PANNs, Whisper).
Output: OUT/finding2_summary.csv, OUT/finding2_per_domain.csv
"""
import numpy as np
import pandas as pd
from scipy import stats

from common import DOMAIN_NAME, OUT, RESULTS, frozen_geometry


def partial(x, y, z):
    rx, ry, rz = (stats.rankdata(v) for v in (x, y, z))
    ex = rx - np.polyval(np.polyfit(rz, rx, 1), rz)
    ey = ry - np.polyval(np.polyfit(rz, ry, 1), rz)
    r = np.corrcoef(ex, ey)[0, 1]
    n = len(x)
    t = r * np.sqrt((n - 3) / (1 - r ** 2))
    return r, 2 * stats.t.sf(abs(t), n - 3)


def main():
    pr = pd.read_csv(RESULTS / "probes" / "frozen_knn_linear.csv")
    tax = pd.read_csv(RESULTS / "models" / "model_arch_taxonomy.csv")[["model", "type"]]
    par = pd.read_csv(RESULTS / "models" / "model_parameter_counts.csv", usecols=[0, 1])
    lc = pd.read_csv(RESULTS / "probes" / "frozen_linear_iterations.csv")[["dataset", "model", "lin_conv"]]
    pr = pr.merge(lc, on=["dataset", "model"], how="left")
    d = pr.merge(frozen_geometry()[["dataset", "model", "uniformity"]], on=["dataset", "model"]).merge(tax, on="model").merge(par, on="model")
    d["discriminative"] = d.type.isin(["Supervised/discriminative", "Cross-modal contrastive"])
    d["reconstruction"] = d.type.str.startswith("Masked")
    d["ratio"] = d.knn / d.linear
    d["gap"] = d.knn - d.lin_conv
    d["log_params"] = np.log10(d.n_params)

    per = []
    for ds, q in d.groupby("dataset"):
        q = q.assign(rank_linear=q.linear.rank(ascending=False), rank_knn=q.knn.rank(ascending=False))
        best_disc = q[q.discriminative].sort_values("rank_linear").iloc[0]
        per.append(dict(domain=DOMAIN_NAME[ds], best_discriminative_model=best_disc.model,
                        its_rank_linear=best_disc.rank_linear, its_rank_knn=best_disc.rank_knn,
                        ratio_ast_clap=q[q.model.isin(["ast", "clap"])].ratio.mean(),
                        ratio_all_supervised_contrastive=q[q.discriminative].ratio.mean(),
                        ratio_reconstruction=q[q.reconstruction].ratio.mean()))
    per = pd.DataFrame(per)
    s = {}
    for col in ["ratio_ast_clap", "ratio_all_supervised_contrastive"]:
        wins = int((per[col] > per.ratio_reconstruction).sum())
        s[f"sign_test_{col}_higher"] = f"{wins}/{len(per)}"
        s[f"sign_test_{col}_p_two_sided"] = stats.binomtest(wins, len(per), 0.5).pvalue
    per.to_csv(OUT / "finding2_per_domain.csv", index=False)

    for ds in ["birdclef", "fma_genre", "mimii"]:
        b = d[(d.dataset == ds) & (d.model == "bird_mae")].iloc[0]
        a = d[(d.dataset == ds) & (d.model == "audiomae")].iloc[0]
        s[f"{DOMAIN_NAME[ds]}_birdmae_over_audiomae_linear"] = b.linear / a.linear
        s[f"{DOMAIN_NAME[ds]}_birdmae_over_audiomae_knn"] = b.knn / a.knn
        s[f"{DOMAIN_NAME[ds]}_birdmae_minus_audiomae_linear"] = b.linear - a.linear
        s[f"{DOMAIN_NAME[ds]}_birdmae_minus_audiomae_knn"] = b.knn - a.knn
    s["n_rows"] = len(d)
    w = d.copy()
    for c in ["uniformity", "knn", "gap", "log_params"]:
        w[c] = w[c] - w.groupby("dataset")[c].transform("mean")
    s["rho_uniformity_knn"], s["p_uniformity_knn"] = stats.spearmanr(w.uniformity, w.knn)
    s["rho_uniformity_gap"], s["p_uniformity_gap"] = stats.spearmanr(w.uniformity, w.gap)
    s["partial_rho_uniformity_knn_given_logparams"], s["partial_p"] = partial(w.uniformity, w.knn, w.log_params)
    r = stats.spearmanr(w.uniformity, w.log_params)[0]
    s["vif_uniformity_logparams"] = 1 / (1 - r ** 2)
    out = pd.Series(s)
    out.to_csv(OUT / "finding2_summary.csv", header=["value"])
    print(per.to_string(index=False))
    print(out.to_string())


if __name__ == "__main__":
    main()
