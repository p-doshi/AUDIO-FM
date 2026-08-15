"""Parameter count vs. accuracy vs. RSA -- does representation geometry
predict downstream accuracy independent of scale, and does cross-model
convergence (mean RSA with the rest of the roster) increase with scale
the way Huh et al. 2024's Platonic Representation Hypothesis claims?

Two checks, both intentionally kept to the simplest form that answers the
question -- correlations/partial correlations with p-values, NOT a fitted
multi-parameter scaling law. At n<=18 models, anything more elaborate
overfits trivially (see the breadth-hypothesis reversal at 18 models for
a concrete example of how sensitive small-N patterns are to exactly which
models are in the roster -- the same caution applies here).

1. Accuracy vs. parameter count, colored by training-objective category
   (discriminative/contrastive/supervised-classification vs. self-
   supervised reconstruction-or-masked-modeling vs. supervised-ASR
   [whisper, a genuinely different kind of supervision signal, kept
   distinct rather than folded into either bucket]). Tests whether
   Stage 4's "discriminative training predicts accuracy" finding
   survives controlling for scale, or whether it was partly standing in
   for "the discriminative models in this roster happen to be bigger."

2. Mean RSA-with-rest-of-roster vs. parameter count. Direct test of Huh
   et al.'s actual scale-convergence claim in the audio domain -- this
   project has cited the paper throughout but never tested this specific
   claim directly until now.

Known limitation, stated explicitly rather than implied away: parameter
count and training-data volume are themselves correlated in most of
these models (bigger models were usually also trained on more data), so
this can show "does scale-and-data-together explain it," not cleanly
separate parameter count from data volume as independent causes -- that
needs actual controlled ablations, out of scope here.

Usage:
    python -m audio_comp.pipelines.scaling_analysis
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

REPO_ROOT = Path(__file__).resolve().parents[2]

# Training-objective category, for plot coloring only -- matches Stage 4's
# own established framing (CLAUDE.md): discriminative training pressure =
# contrastive (clap) or supervised classification with real labels
# (ast, panns_cnn14). Everything else here is self-supervised
# (masked-modeling, data2vec-style self-distillation, reconstruction,
# JEPA-style prediction) -- genuinely heterogeneous internally, but none
# of it is "discriminative" in Stage 4's sense. whisper is its own
# category: ASR/transcript supervision is real supervision, but not the
# same kind of discriminative pressure as class-label prediction or
# contrastive loss -- a judgment call, kept visually distinct rather than
# forced into either bucket.
DISCRIMINATIVE = {"ast", "clap", "panns_cnn14"}
SUPERVISED_ASR = {"whisper"}


def load_param_counts(path: Path) -> dict[str, int]:
    counts = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["n_params"]:
                counts[row["model"]] = int(row["n_params"])
    return counts


def load_accuracy_summary(path: Path) -> dict[str, float]:
    """Mean MLP score across public tasks only (FMA-genre, UrbanSound8K,
    LibriSpeech-100h, BirdCLEF) -- excludes the private DeepShip/ShipsEar
    tasks, which have different class counts/chance floors and would
    distort a cross-model mean if pooled in with the public ones."""
    public_tasks = {
        "Free Music Archive Small",
        "UrbanSound 8k",
        "LibriSpeech-100h",
        "BirdCLEF (50-species, mteb/birdclef25-mini)",
    }
    scores: dict[str, list[float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["task"] in public_tasks:
                scores.setdefault(row["model"], []).append(float(row["mlp_score"]))
    return {model: float(np.mean(vals)) for model, vals in scores.items()}


def load_mean_rsa(path: Path) -> dict[str, float]:
    with open(path) as f:
        reader = csv.reader(f)
        header = next(reader)[1:]
        matrix = np.array([[float(x) for x in row[1:]] for row in reader])
    n = len(header)
    mean_rsa = {}
    for i, model in enumerate(header):
        others = [matrix[i, j] for j in range(n) if j != i]
        mean_rsa[model] = float(np.mean(others))
    return mean_rsa


def category_for(model: str) -> str:
    if model in DISCRIMINATIVE:
        return "discriminative"
    if model in SUPERVISED_ASR:
        return "supervised_asr"
    return "self_supervised"


def spearman_report(x: list[float], y: list[float], label: str) -> None:
    rho, p = spearmanr(x, y)
    print(f"  Spearman {label}: rho={rho:.3f}, p={p:.4f}, n={len(x)}")


def main() -> None:
    params = load_param_counts(REPO_ROOT / "results" / "model_parameter_counts.csv")
    accuracy = load_accuracy_summary(REPO_ROOT / "results" / "xares_results.csv")
    mean_rsa = load_mean_rsa(REPO_ROOT / "results" / "rsa_matrix.csv")

    # === Check 1: accuracy vs params, by objective category ===
    models_acc = sorted(set(params) & set(accuracy))
    print(f"Check 1 (accuracy vs params): {len(models_acc)} models with both metrics: {models_acc}")
    missing_acc = sorted(set(params) - set(accuracy))
    if missing_acc:
        print(f"  (excluded, no public-task accuracy yet: {missing_acc})")

    log_params_acc = [np.log10(params[m]) for m in models_acc]
    acc_vals = [accuracy[m] for m in models_acc]
    cats = [category_for(m) for m in models_acc]
    spearman_report(log_params_acc, acc_vals, "log10(params) vs accuracy")

    is_discriminative = [1 if c == "discriminative" else 0 for c in cats]
    rho_obj, p_obj = spearmanr(is_discriminative, acc_vals)
    print(f"  Spearman discriminative-indicator vs accuracy (unadjusted): rho={rho_obj:.3f}, p={p_obj:.4f}")
    # simple partial-correlation-by-residuals: regress accuracy on
    # log(params), then check whether the *residuals* still correlate
    # with the discriminative indicator -- if they do, the objective
    # effect survives controlling for scale.
    slope, intercept = np.polyfit(log_params_acc, acc_vals, 1)
    residuals = [a - (slope * lp + intercept) for a, lp in zip(acc_vals, log_params_acc)]
    rho_resid, p_resid = spearmanr(is_discriminative, residuals)
    print(f"  Spearman discriminative-indicator vs accuracy RESIDUALS (after removing log(params) trend): "
          f"rho={rho_resid:.3f}, p={p_resid:.4f}")

    fig, ax = plt.subplots(figsize=(8, 6))
    color_map = {"discriminative": "#d62728", "self_supervised": "#1f77b4", "supervised_asr": "#2ca02c"}
    for cat in color_map:
        idx = [i for i, c in enumerate(cats) if c == cat]
        if not idx:
            continue
        ax.scatter(
            [params[models_acc[i]] for i in idx],
            [acc_vals[i] for i in idx],
            c=color_map[cat],
            label=cat,
            s=80,
            edgecolors="black",
            linewidths=0.5,
            zorder=3,
        )
        for i in idx:
            ax.annotate(models_acc[i], (params[models_acc[i]], acc_vals[i]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    trend_x = np.logspace(min(log_params_acc), max(log_params_acc), 50)
    trend_y = slope * np.log10(trend_x) + intercept
    ax.plot(trend_x, trend_y, "--", color="gray", alpha=0.6, zorder=1, label="log-linear trend (all models)")
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("Mean MLP accuracy (public tasks: FMA-genre, UrbanSound8K, LibriSpeech, BirdCLEF)")
    ax.set_title("Accuracy vs. parameter count, by training objective")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "results" / "scaling_accuracy_vs_params.png", dpi=150)
    plt.close(fig)

    # === Check 2: mean RSA vs params (Platonic Representation Hypothesis scale claim) ===
    models_rsa = sorted(set(params) & set(mean_rsa))
    print(f"\nCheck 2 (mean RSA vs params): {len(models_rsa)} models: {models_rsa}")
    log_params_rsa = [np.log10(params[m]) for m in models_rsa]
    rsa_vals = [mean_rsa[m] for m in models_rsa]
    spearman_report(log_params_rsa, rsa_vals, "log10(params) vs mean RSA with rest of roster")

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(
        [params[m] for m in models_rsa], rsa_vals, c="#1f77b4", s=80, edgecolors="black", linewidths=0.5, zorder=3
    )
    for m in models_rsa:
        ax.annotate(m, (params[m], mean_rsa[m]), fontsize=7, xytext=(4, 4), textcoords="offset points")
    slope2, intercept2 = np.polyfit(log_params_rsa, rsa_vals, 1)
    trend_x2 = np.logspace(min(log_params_rsa), max(log_params_rsa), 50)
    ax.plot(trend_x2, slope2 * np.log10(trend_x2) + intercept2, "--", color="gray", alpha=0.6, zorder=1)
    ax.set_xscale("log")
    ax.set_xlabel("Parameter count (log scale)")
    ax.set_ylabel("Mean RSA with every other model in the roster")
    ax.set_title("Representational convergence vs. scale\n(direct test of Huh et al. 2024's scale-convergence claim)")
    fig.tight_layout()
    fig.savefig(REPO_ROOT / "results" / "scaling_rsa_vs_params.png", dpi=150)
    plt.close(fig)

    print(f"\nWrote {REPO_ROOT / 'results' / 'scaling_accuracy_vs_params.png'}")
    print(f"Wrote {REPO_ROOT / 'results' / 'scaling_rsa_vs_params.png'}")


if __name__ == "__main__":
    main()
