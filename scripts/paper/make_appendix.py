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
      "E. Adaptation gain analyses (Finding 3)  \nF. How adaptation reshapes geometry (Finding 3)  \nG. Frozen RSA matrices\n")

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
    (REPO / "APPENDIX.md").write_text("\n".join(S))
    print("wrote", REPO / "APPENDIX.md")


if __name__ == "__main__":
    main()
