"""Accuracy vs. cross-model RSA agreement vs. model size, **per probe-set
category** -- one panel per category, not a single pooled plot.

The earlier v1 of this script used a single global mean-RSA-with-rest-of-
roster number per model (averaged across all 6 probe-set categories) and
plotted it against each model's mean accuracy averaged across public
X-ARES tasks. That conflates two different things and was flagged as
wrong: RSA is inherently a *between-models* quantity that varies by which
clips you compute it on (rsa_by_category.py's whole point), and pairing a
category-agnostic RSA number with a category-agnostic accuracy number
throws away exactly the structure worth looking at -- whether a model's
RSA agreement with the roster *on a given domain's clips* relates to its
*own downstream accuracy on that domain's task*.

v2 instead computes, per category with a matching downstream task:
  x = model's mean RSA with the rest of the roster, computed ONLY on that
      category's clips (reuses rsa_by_category.py's rsa_matrix_for_subset)
  y = model's MLP accuracy on that category's own X-ARES task
  point size = log10(param count)

Category -> task mapping (only categories with a real classification
task are included; `speech` is excluded -- LibriSpeech is ASR, not
classification, no comparable accuracy metric exists):
  music         -> Free Music Archive Small (genre)
  city_noise    -> UrbanSound 8k
  bird_sounds   -> BirdCLEF (50-species, mteb/birdclef25-mini)
  machine_sounds-> MIMII (industrial machine anomaly detection, ...)
  ship_vessel   -> ShipsEar (vessel type A/B/D, real recordings only)
    -- note ShipsEar/DeepShip are the *public* DS3500-derived vessel
    data, not the confidential AIS data; both are already used
    throughout this project's public pipeline and are in scope here.

Usage:
    python -m audio_comp.pipelines.plot_accuracy_vs_rsa_size \
        --embeddings-dir /scratch/pdoshi/audio_comp/embeddings_no_birdnet
"""
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audio_comp.pipelines.rsa_by_category import align_to_common_order, load_categories, load_embeddings
from audio_comp.pipelines.scaling_analysis import category_for, load_param_counts

REPO_ROOT = Path(__file__).resolve().parents[2]
CATEGORY_COLORS = {"discriminative": "#d62728", "self_supervised": "#1f77b4", "supervised_asr": "#2ca02c"}

CATEGORY_TO_TASK = {
    "music": "Free Music Archive Small",
    "city_noise": "UrbanSound 8k",
    "bird_sounds": "BirdCLEF (50-species, mteb/birdclef25-mini)",
    "machine_sounds": "MIMII (industrial machine anomaly detection, normal vs. abnormal, 6dB tier)",
    "ship_vessel": "ShipsEar (vessel type A/B/D, real recordings only)",
}


def load_mlp_scores(path: Path) -> dict[str, dict[str, float]]:
    """task -> {model: mlp_score}."""
    scores: dict[str, dict[str, float]] = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            scores.setdefault(row["task"], {})[row["model"]] = float(row["mlp_score"])
    return scores


def mean_rsa_within_subset(aligned: dict[str, np.ndarray], model_names: list[str], row_idx: np.ndarray) -> dict[str, float]:
    from audio_comp.geometry.rdm import compute_rdm
    from audio_comp.geometry.rsa import rsa_score

    rdms = {name: compute_rdm(aligned[name][row_idx]) for name in model_names}
    mean_rsa = {}
    for name in model_names:
        others = [rsa_score(rdms[name], rdms[other]) for other in model_names if other != name]
        mean_rsa[name] = float(np.mean(others))
    return mean_rsa


def main(embeddings_dir: Path, manifest_path: Path, out_path: Path) -> None:
    embeddings = load_embeddings(embeddings_dir)
    model_names = sorted(embeddings.keys())
    reference_ids, aligned = align_to_common_order(embeddings, model_names)
    clip_to_category = load_categories(manifest_path)
    ref_categories = np.array([clip_to_category[cid] for cid in reference_ids])

    params = load_param_counts(REPO_ROOT / "results" / "model_parameter_counts.csv")
    mlp_scores = load_mlp_scores(REPO_ROOT / "results" / "xares_results.csv")

    categories = [c for c in CATEGORY_TO_TASK if c in set(ref_categories)]
    ncols = 3
    nrows = math.ceil(len(categories) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(7 * ncols, 6 * nrows))
    axes = np.atleast_1d(axes).flatten()

    for ax, category in zip(axes, categories):
        task = CATEGORY_TO_TASK[category]
        row_idx = np.where(ref_categories == category)[0]
        mean_rsa = mean_rsa_within_subset(aligned, model_names, row_idx)
        task_scores = mlp_scores.get(task, {})

        models = sorted(set(model_names) & set(task_scores) & set(params))
        if not models:
            ax.set_title(f"{category}\n(no models with all 3 metrics)")
            ax.axis("off")
            continue

        xs = [mean_rsa[m] for m in models]
        ys = [task_scores[m] for m in models]
        log_params = [np.log10(params[m]) for m in models]
        lo, hi = min(log_params), max(log_params)
        sizes = [150 + 1200 * (lp - lo) / (hi - lo) if hi > lo else 400 for lp in log_params]
        colors = [CATEGORY_COLORS[category_for(m)] for m in models]

        ax.scatter(xs, ys, s=sizes, c=colors, alpha=0.7, edgecolors="black", linewidths=0.7)
        for m, x, y in zip(models, xs, ys):
            ax.annotate(m, (x, y), fontsize=6, xytext=(3, 3), textcoords="offset points")
        ax.set_xlabel("Mean within-category RSA with rest of roster")
        ax.set_ylabel("MLP accuracy (this category's task)")
        ax.set_title(f"{category}\n({task}, n={len(row_idx)} clips, {len(models)} models)", fontsize=10)

    for ax in axes[len(categories):]:
        ax.axis("off")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=10, label=cat)
        for cat, c in CATEGORY_COLORS.items()
    ]
    fig.legend(handles=handles, title="training objective", loc="lower right")
    fig.suptitle(
        "Per-category: downstream accuracy vs. within-category cross-model RSA vs. model size\n"
        "(point size = log10 param count; RSA computed only on that category's own clips)",
        fontsize=13,
    )
    fig.subplots_adjust(hspace=0.45, wspace=0.35, top=0.88)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "probe_set_manifest.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "accuracy_vs_rsa_vs_size_by_category.png")
    args = parser.parse_args()
    main(args.embeddings_dir, args.manifest, args.out)
