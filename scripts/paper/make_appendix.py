"""Build APPENDIX.md and its figures from results/ (aggregate inputs) and results/paper_outputs/ (outputs of the
finding*.py scripts). Run after the analysis scripts:
    python finding1.py; python finding2.py; python finding3_gain_regression.py; python finding3_predictor_search.py
    python finding3_transferability.py; python finding3_reorganization.py; python make_appendix.py
"""
import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import (DOMAIN_NAME, DOMAINS, LORA_MODELS, OUT, REPO, RESULTS, TOPK_MODELS, frozen_geometry,  # noqa: E402
                    load_converged_probe, read_square)

sys.path.insert(0, str(REPO))
from audio_comp.pipelines.breadth_hypothesis_check import BREADTH_GROUPS, DOMAIN_PARADIGM_GROUPS  # noqa: E402

FIG = REPO / "figures" / "appendix"
FIG.mkdir(parents=True, exist_ok=True)
NAME = {"ast": "AST", "audio_jepa": "Audio-JEPA", "audiomae": "AudioMAE", "bird_mae": "Bird-MAE", "clap": "CLAP",
        "data2vec_audio": "data2vec-audio", "encodecmae": "EnCodecMAE", "hubert": "HuBERT", "mert": "MERT", "mms": "MMS",
        "music2vec": "music2vec", "musicfm": "MusicFM", "panns_cnn14": "PANNs (CNN14)", "sew": "SEW",
        "unispeech_sat": "UniSpeech-SAT", "wav2vec2": "wav2vec 2.0", "wav2vec2_conformer": "wav2vec 2.0 Conformer",
        "wavlm": "WavLM", "whisper": "Whisper"}
CKPT = {"ast": "MIT/ast-finetuned-audioset-10-10-0.4593", "panns_cnn14": "Zenodo 3987831, Cnn14_mAP=0.431.pth",
        "whisper": "openai/whisper-base", "clap": "laion/larger_clap_general",
        "audiomae": "facebookresearch/AudioMAE, pretrained.pth (ViT-B/16, README-linked Google Drive)", "bird_mae": "DBD-research-group/Bird-MAE-Base",
        "encodecmae": "habla-liaa/encodecmae, model ec-ec-large_st", "hubert": "facebook/hubert-large-ll60k",
        "wav2vec2": "facebook/wav2vec2-large-lv60", "wav2vec2_conformer": "facebook/wav2vec2-conformer-rel-pos-large",
        "wavlm": "microsoft/wavlm-base-plus", "unispeech_sat": "microsoft/unispeech-sat-base", "sew": "asapp/sew-tiny-100k",
        "mms": "facebook/mms-300m", "mert": "m-a-p/MERT-v1-330M", "musicfm": "minzwon/MusicFM, pretrained_msd.pt",
        "audio_jepa": "ltuncay/Audio-JEPA, JEPA.ckpt", "data2vec_audio": "facebook/data2vec-audio-base", "music2vec": "m-a-p/music2vec-v1"}
ORDER = list(NAME)
f2, f3 = lambda v: f"{v:.2f}", lambda v: f"{v:.3f}"


def md(df, floatfmt=".3f"):
    return df.to_markdown(index=False, floatfmt=floatfmt)


def roster():
    tax = pd.read_csv(RESULTS / "models" / "model_arch_taxonomy.csv")[["model", "type"]]
    par = pd.read_csv(RESULTS / "models" / "model_parameter_counts.csv", usecols=[0, 1])
    d = tax.merge(par, on="model")
    d["adaptation"] = d.model.map(lambda m: "LoRA, ALLoRA" if m in LORA_MODELS else "top-K" if m in TOPK_MODELS else "frozen only")
    d["checkpoint"] = d.model.map(CKPT)
    d["params (M)"] = (d.n_params / 1e6).round(1)
    d["model"] = pd.Categorical(d.model, ORDER)
    d = d.sort_values("model")
    d["model"] = d.model.map(NAME)
    return md(d[["model", "type", "params (M)", "adaptation", "checkpoint"]].rename(columns={"type": "training paradigm"}), ".1f")


def accuracy_table():
    p = load_converged_probe()
    p = p[p.condition == p.model.map(lambda m: "topk" if m in TOPK_MODELS else "lora")]
    al = load_converged_probe(allora=True)
    out = []
    for ds in DOMAINS:
        for m in ORDER:
            a = p[(p.dataset == ds) & (p.model == m)]
            b = al[(al.dataset == ds) & (al.model == m)]
            if a.empty and b.empty:
                continue
            fr = pd.concat([a.frozen_acc, b.frozen_acc])
            row = dict(domain=DOMAIN_NAME[ds], model=NAME[m], seeds=len(a) or len(b),
                       frozen=f"{fr.mean():.3f} ± {fr.std(ddof=1) if len(fr) > 1 else 0:.3f}")
            cond = "top-K" if m in TOPK_MODELS else "LoRA"
            row["adapted (primary)"] = f"{a.adapted_acc.mean():.3f} ± {a.adapted_acc.std(ddof=1):.3f} ({cond})" if len(a) else "excluded"
            row["gain (primary)"] = f"{a.gain.mean():+.3f}" if len(a) else ""
            row["adapted (ALLoRA)"] = f"{b.adapted_acc.mean():.3f} ± {b.adapted_acc.std(ddof=1):.3f}" if len(b) > 1 else ("" if b.empty else f"{b.adapted_acc.mean():.3f}")
            out.append(row)
    return md(pd.DataFrame(out))


def frozen_table():
    pr = pd.read_csv(RESULTS / "probes" / "frozen_knn_linear.csv")
    lc = pd.read_csv(RESULTS / "probes" / "frozen_linear_iterations.csv")[["dataset", "model", "lin_conv"]]
    g = frozen_geometry()[["dataset", "model", "uniformity", "alignment", "twonn"]]
    d = pr.merge(lc, on=["dataset", "model"]).merge(g, on=["dataset", "model"], how="left")
    d["domain"] = d.dataset.map(DOMAIN_NAME)
    d["model"] = pd.Categorical(d.model, ORDER)
    d = d.sort_values(["domain", "model"])
    d["model"] = d.model.map(NAME)
    return md(d[["domain", "model", "n_test", "linear", "lin_conv", "knn", "uniformity", "alignment", "twonn"]]
              .rename(columns={"lin_conv": "linear (converged)", "twonn": "TwoNN"}))


def rsa_heatmaps():
    files = []
    for ds in DOMAINS:
        M = read_square(RESULTS / "geometry" / "rsa_frozen" / f"rsa_{ds}.csv")
        order = [m for m in ORDER if m in M.index]
        M = M.loc[order, order]
        fig, ax = plt.subplots(figsize=(7.2, 6.2))
        v = M.values.copy()
        off = v[~np.eye(len(v), dtype=bool)]
        im = ax.imshow(v, cmap="Blues", vmin=off.min(), vmax=off.max())
        ax.set_xticks(range(len(order)), [NAME[m] for m in order], rotation=90, fontsize=8)
        ax.set_yticks(range(len(order)), [NAME[m] for m in order], fontsize=8)
        ax.set_title(f"Frozen RSA between models, {DOMAIN_NAME[ds]} (off-diagonal range {off.min():.2f} to {off.max():.2f})", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="RSA (Spearman)")
        fig.tight_layout()
        f = FIG / f"rsa_frozen_{DOMAIN_NAME[ds]}.png"
        fig.savefig(f, dpi=200)
        plt.close(fig)
        files.append(f)
    return files


def groups_table():
    rows = []
    for name, G in [("domain + paradigm", DOMAIN_PARADIGM_GROUPS), ("training-data breadth", BREADTH_GROUPS)]:
        for g, ms in G.items():
            ms = [NAME[m] for m in ms if m in NAME]
            if ms:
                rows.append(dict(grouping=name, group=g, models=", ".join(ms)))
    return md(pd.DataFrame(rows))


def opt(path):
    return pd.read_csv(path) if Path(path).exists() else None


def robustness_section(w):
    R = RESULTS / "robustness"
    if not (R / "probes_robust.csv").exists():
        return
    w("## H. Robustness checks\n")
    w("All checks use the frozen embeddings of the five public domains (`scripts/paper/robustness_frozen.py`, analysis in "
      "`scripts/paper/robustness_analysis.py`). Splits are those of §C. The MIMII wav2vec 2.0 Conformer cell, absent from "
      "the Finding 2 probe table, is present here, so n = 95 (19 models × 5 domains) instead of 94.\n")

    w("### H.1 Uniformity and frozen kNN accuracy across kNN settings\n")
    w("kNN with k ∈ {1, 5, 10, 20, 50} under two metrics: Euclidean distance on standardized features (the setting of "
      "Finding 2) and cosine similarity on raw embeddings. Scores: accuracy and balanced (macro-averaged) accuracy. Gap = "
      "kNN − converged linear probe, both scored the same way. ρ: Spearman after demeaning within domain (uniformity at "
      "t = 2). p (model perm.): two-sided, model identities permuted jointly in all domains (2,000 draws; the smallest "
      "attainable value is 0.0005). p (scipy) treats the 95 points as independent and is shown for reference.\n")
    u = pd.read_csv(OUT / "robust_uniformity_knn.csv")
    a = u[u.vary == "kNN"].copy()
    a["setting"] = a.metric + ", k = " + a.k.astype(str)
    piv = a.pivot_table(index=["metric", "k"], columns=["score", "target"], values=["rho", "p_model_perm"])
    piv.columns = [f"{v} {sc} {tg}".replace("p_model_perm", "p").replace("rho", "ρ") for v, sc, tg in piv.columns]
    order = [f"{v} {sc} {tg}" for sc in ["acc", "bal_acc"] for tg in ["knn", "gap"] for v in ["ρ", "p"]]
    w(md(piv[order].reset_index(), ".4f") + "\n")
    w("### H.2 Uniformity temperature\n")
    w("Uniformity log E[exp(−t‖z_i − z_j‖²)] over all clip pairs of L2-normalized embeddings, t ∈ {0.5, 1, 2, 5, 10} "
      "(the paper uses t = 2). kNN: Euclidean on standardized features, k = 10, accuracy.\n")
    b = u[u.vary == "t"].pivot_table(index="t", columns="target", values=["rho", "p_model_perm", "p_scipy"])
    b.columns = [f"{tg} {v}".replace("p_model_perm", "p (model perm.)").replace("p_scipy", "p (scipy)").replace("rho", "ρ") for v, tg in b.columns]
    w(md(b.reset_index(), ".4f") + "\n")

    w("### H.3 Per-model values\n")
    w("kNN accuracy (k = 10) under both metrics, balanced accuracy of the Finding 2 kNN and of the converged linear probe, "
      "uniformity at each temperature, and the two additional transferability estimators (§H.4).\n")
    pr = pd.read_csv(R / "probes_robust.csv")
    geo = pd.read_csv(R / "geo_te_robust.csv")
    k10 = pr[(pr.probe == "knn") & (pr.k == 10)].pivot_table(index=["dataset", "model"], columns="metric", values=["acc", "bal_acc"])
    k10.columns = [f"kNN {m} {s}".replace("euclidean_std", "eucl.").replace("cosine_raw", "cos.").replace("bal_acc", "bal. acc") for s, m in k10.columns]
    lin = pr[pr.probe == "linear_conv"].set_index(["dataset", "model"])[["acc", "bal_acc"]].rename(columns={"acc": "linear acc", "bal_acc": "linear bal. acc"})
    t = k10.join(lin).join(geo.set_index(["dataset", "model"]).drop(columns="n")).reset_index()
    t.insert(0, "domain", t.dataset.map(DOMAIN_NAME))
    t["model"] = t.model.map(NAME)
    w(md(t.drop(columns="dataset"), ".3f") + "\n")
    w("Accuracy for every k and metric is in `results/robustness/probes_robust.csv`.\n")

    w("### H.4 Additional transferability estimators\n")
    w("Shrinkage H-score (Ibrahim et al., ECML PKDD 2022): tr(Σ_α⁻¹ (1 − α) Σ_between), with Σ_α the Ledoit–Wolf shrinkage "
      "estimate of the feature covariance and Σ_between the covariance of the class-mean features. GBC (Pándy et al., CVPR "
      "2022): −Σ over class pairs of the Bhattacharyya coefficient between per-class diagonal Gaussians fitted on the first "
      "64 principal components. Both use frozen embeddings and labels of all clips. Tested against seed-averaged "
      "converged-probe gain exactly as LogME (§E): within-domain Spearman, two-sided model-identity permutation (5,000 "
      "draws), 95% CI from 2,000 model bootstraps; 90 (model, domain) pairs.\n")
    e = pd.concat([pd.read_csv(OUT / "finding3_transferability.csv"), pd.read_csv(OUT / "robust_transferability.csv")])
    w(md(e, ".4f") + "\n")

    w("### H.5 Cross-domain stability under other similarity measures\n")
    w("The Finding 1 test (Spearman between two domains' model × model matrices, one-sided model-identity permutation, "
      "10,000 permutations, BH over the 10 public pairs) for linear CKA, debiased linear CKA (unbiased HSIC estimator, "
      "Song et al., 2012) and mutual k-NN overlap (fraction of shared 10 nearest neighbours per clip, cosine, averaged "
      "over clips), next to RSA. The vessel domain is not included. MIMII lacks the Conformer RSA row, so the RSA "
      "comparison uses 18 models.\n")
    c = pd.read_csv(OUT / "robust_cross_domain.csv")
    pc = c.pivot_table(index=["domain_a", "domain_b"], columns="metric", values=["rho", "q_bh"])
    pc.columns = [f"{m} {v}".replace("q_bh", "q").replace("rho", "ρ") for v, m in pc.columns]
    w(md(pc[[f"{m} {v}" for m in ["rsa", "cka", "dcka", "mknn"] for v in ["ρ", "q"]]].reset_index(), ".4f") + "\n")
    diff = (pc["cka ρ"] - pc["dcka ρ"]).abs().max()
    r2 = []
    for d in DOMAINS:
        M = read_square(R / f"sim_mknn_{d}.csv").values.astype(float)
        iu = np.triu_indices(len(M), 1)
        r = np.array([np.mean(np.delete(M[i], i)) for i in range(len(M))])
        r2.append(np.corrcoef((r[:, None] + r[None, :])[iu], M[iu])[0, 1] ** 2)
    w(f"Cross-domain ρ under debiased and standard linear CKA differ by at most {diff:.3f} in any pair, so finite-sample "
      "bias does not change the comparison at these sample sizes (2,400 to 10,026 clips). Mutual k-NN overlap makes "
      "every pair significant, including the speech pairs; an additive per-model term (each model's mean similarity to "
      f"the others) explains {100 * min(r2):.0f}–{100 * max(r2):.0f}% of the variance of each mutual k-NN matrix.\n")
    d = opt(OUT / "robust_cross_domain_model_effect_removed.csv")
    if d is not None:
        w("### H.6 Exploratory: model main effect removed\n")
        w("Each matrix is double-centred (row and column means removed, diagonal excluded) before the same test, so only "
          "pair-specific affinity beyond each model's overall level of agreement remains. Centrality: Spearman across "
          "models between two domains' per-model mean similarity to the other models.\n")
        pd2 = d.pivot_table(index=["domain_a", "domain_b"], columns="metric", values=["rho", "q_bh"])
        pd2.columns = [f"{m} {v}".replace("q_bh", "q").replace("rho", "ρ") for v, m in pd2.columns]
        ce = pd.read_csv(OUT / "robust_centrality.csv").pivot_table(index=["domain_a", "domain_b"], columns="metric", values="rho_centrality")
        ce.columns = [f"{m} centrality ρ" for m in ce.columns]
        cols = [f"{m} {v}" for m in ["rsa", "dcka", "mknn"] for v in ["ρ", "q"]]
        w(md(pd2[cols].join(ce, how="left").reset_index(), ".4f") + "\n")
        w("With the per-model term removed, all 10 pairs are significant under every measure, including the speech pairs. "
          "What differs in speech is which models are central: per-model centrality in speech correlates weakly with the "
          "other domains, with data2vec-audio, music2vec and wav2vec 2.0 more central and AudioMAE, EnCodecMAE, CLAP and "
          "Bird-MAE less central than elsewhere.\n")

    M = R / "mlp_head"
    if (M / "agreement.csv").exists():
        w("### H.7 Non-linear probe head (MLP)\n")
        w("Adaptation gain recomputed with a one-hidden-layer MLP (512 ReLU units, Adam, early stopping, L2 penalty chosen "
          "on validation from {1e-4, 1e-2, 1}) in place of the linear probe, on the same Gram-recovered features, splits, "
          "conditions and checkpoints (`scripts/paper/mlp_converged_gain.py`; analysis `scripts/paper/mlp_head_analysis.py`). "
          "\"white\": the whitened features of the paper; \"scaled\": the embeddings up to rotation with one global scale. "
          "Per-cell values: `results/robustness/mlp_gain.csv`.\n")
        w("Agreement with the linear probe (Spearman; pair = seed-averaged (model, domain) gain; within domain = mean over "
          "the five domains):\n")
        w(md(pd.read_csv(M / "agreement.csv"), ".3f") + "\n")
        for tag in ["white", "scaled"]:
            reg = pd.read_csv(M / f"{tag}_finding3_regression.csv")
            w(f"Finding 3 regression with MLP gain ({tag} features), as in §E:\n")
            w(md(reg, ".4f") + "\n")
            tr = pd.concat([pd.read_csv(M / f"{tag}_finding3_transferability.csv"), pd.read_csv(M / f"{tag}_robust_transferability.csv")])
            w(f"Transferability estimators vs MLP gain ({tag} features), as in §E and §H.4:\n")
            w(md(tr, ".4f") + "\n")
        reg = pd.read_csv(M / "white_finding3_regression.csv").set_index("variant").iloc[0]
        tr = pd.concat([pd.read_csv(M / "white_finding3_transferability.csv"), pd.read_csv(M / "white_robust_transferability.csv")])
        w(f"With the MLP head the regression explains R² = {reg.R2:.3f} of the variance in gain (whitened features); "
          f"domain ease is the only term with p < 0.05 (p = {reg.p_difficulty:.3f}; uniformity p = {reg.p_uniformity:.2f}), "
          f"and no estimator predicts gain within domain (|ρ| ≤ {tr.within_domain_rho.abs().max():.2f}, "
          f"smallest p = {tr.perm_p.min():.2f}).\n")
    targets_section(w)
    layerwise_section(w)


def targets_section(w):
    t = opt(OUT / "robust_gain_targets.csv")
    if t is not None:
        w("### H.8 Transferability estimators: adapted accuracy versus gain\n")
        w("Eight frozen estimators against four seed-averaged targets per (model, domain): gain (adapted − frozen accuracy, "
          "the target of Finding 3), normalized gain (adapted − frozen) / (1 − frozen), adapted accuracy, and frozen "
          "accuracy. Besides LogME, kNN with labelled reference clips, shrinkage H-score and GBC (§E, §H.4): SFDA (Shao et "
          "al., ECCV 2022) and NCTI (Wang et al., ICCV 2023), both with the authors' released code, and TransRate (Huang et "
          "al., ICML 2022; ε = 1e-4, centred L2-normalized embeddings) (`scripts/paper/transferability_more.py`; values in "
          "`results/robustness/transferability_more.csv`). NCTI's three terms are min–max normalized across the models of "
          "each domain, as in the released evaluation script. Statistic: within-domain Spearman over 90 (model, domain) "
          "pairs; p: two-sided model-identity permutation (5,000 draws); CI: 2,000 model bootstraps "
          "(`scripts/paper/robustness_targets.py`, part A).\n")
        piv = t.pivot_table(index="estimator", columns="target", values=["within_domain_rho", "perm_p"], sort=False)
        piv.columns = [f"{tg} {'ρ' if v == 'within_domain_rho' else 'p'}" for v, tg in piv.columns]
        cols = [f"{tg} {v}" for tg in ["gain", "gain_norm", "adapted_acc", "frozen_acc"] for v in ["ρ", "p"]]
        w(md(piv[cols].reset_index(), ".3f") + "\n")
        a, g, n = (t[t.target == x] for x in ["adapted_acc", "gain", "gain_norm"])
        sig = a[a.perm_p < 0.05].sort_values("within_domain_rho", ascending=False)
        w(f"{len(sig)} of {len(a)} estimators rank adapted accuracy within domain (p < 0.05; ρ = "
          f"{sig.within_domain_rho.min():.2f}–{sig.within_domain_rho.max():.2f}), so they behave as designed on this "
          f"benchmark, yet none predicts gain (largest |ρ| = {g.within_domain_rho.abs().max():.2f}, smallest p = "
          f"{g.perm_p.min():.2f}) or normalized gain (largest |ρ| = {n.within_domain_rho.abs().max():.2f}, smallest p = "
          f"{n.perm_p.min():.2f}). The estimators track the level a model reaches, which is mostly the level it starts "
          "from (frozen accuracy column), not the change produced by adaptation. Full table with CIs: "
          "`results/paper_outputs/robust_gain_targets.csv`.\n")
    r = opt(OUT / "robust_regression_variants.csv")
    if r is not None:
        w("### H.9 Regression variants\n")
        w("gain ~ difficulty + uniformity + domain match as in §E (model-clustered SE, t reference with G − 1 = 17 df), with "
          "difficulty computed leaving each row's own model out, and with normalized gain (adapted − frozen) / (1 − frozen) "
          "as the target.\n")
        w(md(r, ".4f") + "\n")
    s = opt(OUT / "robust_cross_domain_subsets.csv")
    if s is not None:
        w("### H.10 Cross-domain stability within pretraining cohorts\n")
        w("The Finding 1 test restricted to the 10 encoders not pretrained on speech and to the 9 speech-pretrained "
          "encoders (HuBERT, wav2vec 2.0, WavLM, UniSpeech-SAT, SEW, wav2vec 2.0 Conformer, MMS, data2vec-audio, Whisper), "
          "for RSA and linear CKA; BH over the 10 public pairs within each subset and measure.\n")
        p = s.pivot_table(index=["domain_a", "domain_b"], columns=["metric", "subset"], values=["rho", "q_bh"], sort=False)
        p.columns = [f"{m} {sub} {'ρ' if v == 'rho' else 'q'}" for v, m, sub in p.columns]
        cols = [f"{m} {sub} {v}" for m in ["rsa", "cka"] for sub in ["not speech-pretrained", "speech-pretrained"] for v in ["ρ", "q"]]
        w(md(p[cols].reset_index(), ".3f") + "\n")
        rs = s[(s.metric == "rsa")]
        sp = rs[(rs.domain_a == "speech") | (rs.domain_b == "speech")]
        ns = rs[(rs.domain_a != "speech") & (rs.domain_b != "speech")]
        ck = s[(s.metric == "cka") & (s.subset == "speech-pretrained") & ((s.domain_a == "speech") | (s.domain_b == "speech"))]
        w(f"Under RSA no speech pair survives correction in either cohort (q ≥ {sp.q_bh.min():.2f}), so the RSA speech "
          "reorganization is not produced by the speech-pretrained models alone. With 9 or 10 models the test has less "
          f"power: {int((ns.q_bh >= 0.05).sum())} of the {len(ns)} non-speech RSA cells (two cohorts × six pairs) also miss "
          f"q < 0.05. Under linear CKA the speech-pretrained cohort agrees with speech in {int((ck.q_bh < 0.05).sum())} of "
          f"{len(ck)} speech pairs, so for these models the CKA view of speech is closer to the other domains than the RSA "
          "view.\n")
    d = opt(OUT / "robust_ratio_groups.csv")
    if d is not None:
        w("### H.11 kNN/linear ratio, discriminative versus reconstruction, at model level\n")
        w("Replaces the five-domain sign test of `finding2.py` with a model-level test: kNN/linear accuracy ratio demeaned "
          "within domain, difference between the group means over all (model, domain) points, two-sided permutation of the "
          "group labels across models (same in every domain, 10,000 draws). Reconstruction = the 12 masked-modelling "
          "encoders.\n")
        w(md(d, ".4f") + "\n")


def layerwise_section(w):
    R = RESULTS / "robustness"
    if not (R / "layerwise_probes.csv").exists():
        return


def main():
    S = []
    w = S.append
    w("# ECHO: appendix\n")
    w("This appendix accompanies the paper. Every table below is generated by `scripts/paper/make_appendix.py` from the "
      "aggregate results in `results/`; the analysis code is in `scripts/paper/` and the experimental specification "
      "(datasets, splits, hyper-parameters, metrics, statistical tests) is in [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md). "
      "Vessel results are aggregate statistics only; no vessel audio, labels or per-model vessel values are released.\n")
    w("## Contents\n\nA. Model roster and checkpoints  \nB. Frozen and adapted accuracy (converged probe)  \n"
      "C. Frozen usability probes and geometry (Finding 2)  \nD. Cross-domain stability and taxonomy tests (Finding 1)  \n"
      "E. Adaptation gain analyses (Finding 3)  \nF. How adaptation reshapes geometry (Finding 3)  \nG. Frozen RSA matrices  \n"
      "H. Robustness checks\n")

    w("## A. Model roster and checkpoints\n")
    w("Pretrained checkpoints used for every model (Hugging Face identifiers unless stated). No adapted weights are "
      "distributed: the adaptation pipeline stores only clip × clip RDM and Gram matrices (§4 of REPRODUCIBILITY.md).\n")
    w(roster() + "\n")

    w("## B. Frozen and adapted accuracy (converged probe)\n")
    w("Test accuracy of the converged linear probe on Gram-recovered features of the frozen and the adapted encoder, mean ± SD "
      "over seeds. Primary condition: LoRA for the 14 LoRA-eligible models, top-K for the other 4. Gain = adapted − frozen "
      "(primary condition). Speech LoRA misses some seeds.\n")
    w(accuracy_table() + "\n")

    w("## C. Frozen usability probes and geometry (Finding 2)\n")
    w("Linear: logistic regression (C = 1) on standardized frozen embeddings; linear (converged): the same model run to "
      "convergence, used for the kNN − linear gap; kNN: k = 10. Geometry columns are computed on each domain's full pool.\n")
    w(frozen_table() + "\n")
    s = opt(OUT / "finding2_summary.csv")
    if s is not None:
        w("Summary statistics (`finding2.py`):\n")
        w(md(s.rename(columns={"Unnamed: 0": "statistic"}), ".4f") + "\n")
    s = opt(OUT / "finding2_per_domain.csv")
    if s is not None:
        w("kNN/linear accuracy ratio by domain:\n")
        w(md(s) + "\n")

    w("## D. Cross-domain stability and taxonomy tests (Finding 1)\n")
    x = pd.read_csv(RESULTS / "finding1" / "cross_domain_stability.csv")
    w("Spearman ρ between two domains' 19 × 19 frozen RSA matrices (171 model pairs), one-sided model-identity permutation "
      "test (10,000 permutations), BH over the 15 pairs. LOO: range of ρ when each model is removed.\n")
    w(md(x[["a", "b", "rho", "p_perm", "q_bh", "loo_min", "loo_max", "loo_min_model"]]
         .rename(columns={"a": "domain A", "b": "domain B", "loo_min_model": "model whose removal gives the minimum"}), ".4f") + "\n")
    w("Leave-one-model-out with BH within each roster (5,000 permutations): no removal makes a speech pair significant; "
      "removing AST or wav2vec 2.0 drops city–machine below significance and removing data2vec-audio drops city–vessel; "
      "every other removal leaves all 11 pairs significant (`results/finding1/cross_domain_leave_one_out.txt`).\n")
    w("![Cross-domain stability](figures/F1_xdomain.png)\n")
    t = pd.read_csv(RESULTS / "finding1" / "taxonomy_mantel.csv")
    w("Taxonomy alignment: Mantel r between each domain's RSA matrix and a same-group indicator, one-sided permutation p "
      "(200,000 permutations), 95% bootstrap CI over models. None of the 12 tests survives BH at q = 0.05.\n")
    w(md(t[["domain", "taxonomy", "n_models", "mantel_r", "p_perm", "ci_lo", "ci_hi", "bh_survives"]], ".4f") + "\n")
    w("Group definitions (`audio_comp/pipelines/breadth_hypothesis_check.py`; models not listed form singleton groups):\n")
    w(groups_table() + "\n")

    w("## E. Adaptation gain analyses (Finding 3)\n")
    for f, title in [("finding3_regression.csv", "Regression gain ~ difficulty + uniformity + domain match (model-clustered SE; R² CI and within-domain ρ CI from 2,000 model bootstraps)"),
                     ("finding3_predictor_search.csv", "Predictor search (leave-one-model-out CV R², permutation p over the full search, 2,000 permutations)"),
                     ("finding3_transferability.csv", "Transferability estimators vs gain (within-domain Spearman, permutation p, model-bootstrap CI)")]:
        d = opt(OUT / f)
        if d is not None:
            w(f"{title}:\n")
            w(md(d, ".4f") + "\n")

    w("## F. How adaptation reshapes geometry (Finding 3)\n")
    for f, title in [("finding3_delta_rsa.csv", "ΔRSA (mean over models of the change in mean RSA to the other models; WavLM excluded) and preserved ordering (Spearman between frozen and adapted mean RSA, permutation p, BH over 10 cells)"),
                     ("finding3_self_rsa_links.csv", "Self-RSA links at model level (seeds averaged; gain = converged-probe gain of that condition)")]:
        d = opt(OUT / f)
        if d is not None:
            w(f"{title}:\n")
            w(md(d, ".4f") + "\n")
    d = opt(OUT / "finding3_shared_offset.csv")
    if d is not None:
        w(f"Shared offset (‖mean of L2-normalized embeddings‖²) falls in {int(d.falls.sum())} of {len(d)} valid adaptation runs; "
          f"exceptions: {', '.join(f'{DOMAIN_NAME[r.dataset]} {NAME[r.model]} {r.condition} seed {r.seed}' for r in d[~d.falls].itertuples())}.\n")

    w("## G. Frozen RSA matrices\n")
    for f in rsa_heatmaps():
        w(f"![{f.stem}](figures/appendix/{f.name})\n")

    robustness_section(w)
    (REPO / "APPENDIX.md").write_text("\n".join(S))
    print("wrote", REPO / "APPENDIX.md")


if __name__ == "__main__":
    main()
