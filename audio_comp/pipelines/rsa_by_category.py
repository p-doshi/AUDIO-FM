"""Per-category model-vs-model RSA intercorrelation grid -- same idea as
brain_rsa/plots/rsa_intercorrelation_by_category.png, but for this
project's own probe set/model roster (no brain data, no per-layer depth
search -- just final-layer embeddings, one RSA matrix per probe-set
category). v1: quick look, not yet interpreted.

Usage:
    python -m audio_comp.pipelines.rsa_by_category \
        --embeddings-dir $SCRATCH/audio_comp/embeddings \
        --manifest data/probe_set_manifest.csv \
        --out results/rsa_by_category.png
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

from audio_comp.geometry.rdm import compute_rdm
from audio_comp.geometry.rsa import rsa_score

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_embeddings(embeddings_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for path in sorted(embeddings_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        result[path.stem] = (data["clip_ids"], data["embeddings"])
    if not result:
        raise RuntimeError(f"no .npz embedding files found in {embeddings_dir}")
    return result


def load_categories(manifest_path: Path) -> dict[str, str]:
    with open(manifest_path, newline="") as f:
        return {row["clip_id"]: row["category"] for row in csv.DictReader(f)}


def align_to_common_order(
    embeddings: dict[str, tuple[np.ndarray, np.ndarray]], model_names: list[str]
) -> tuple[list[str], dict[str, np.ndarray]]:
    reference_ids = list(embeddings[model_names[0]][0])
    aligned = {}
    for name in model_names:
        clip_ids, embeds = embeddings[name]
        id_to_row = {cid: i for i, cid in enumerate(clip_ids)}
        missing = [cid for cid in reference_ids if cid not in id_to_row]
        if missing:
            raise RuntimeError(f"model '{name}' is missing {len(missing)} clip(s) present in the reference set")
        order = [id_to_row[cid] for cid in reference_ids]
        aligned[name] = embeds[order]
    return reference_ids, aligned


def rsa_matrix_for_subset(aligned: dict[str, np.ndarray], model_names: list[str], row_idx: np.ndarray) -> np.ndarray:
    rdms = {name: compute_rdm(aligned[name][row_idx]) for name in model_names}
    n = len(model_names)
    matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            score = rsa_score(rdms[model_names[i]], rdms[model_names[j]])
            matrix[i, j] = matrix[j, i] = score
    return matrix


def main(embeddings_dir: Path, manifest_path: Path, out_path: Path) -> None:
    embeddings = load_embeddings(embeddings_dir)
    model_names = sorted(embeddings.keys())
    print(f"{len(model_names)} models: {model_names}")

    reference_ids, aligned = align_to_common_order(embeddings, model_names)
    clip_to_category = load_categories(manifest_path)
    categories = sorted(set(clip_to_category.values()))
    print(f"categories: {categories}")

    ref_categories = np.array([clip_to_category[cid] for cid in reference_ids])

    ncols = 3
    nrows = math.ceil(len(categories) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(6.5 * ncols, 7 * nrows))
    axes = np.atleast_1d(axes).flatten()

    # Color scale tightened to the actual observed data range (not the
    # full theoretical -1..1) so relative differences are visible instead
    # of everything reading as one uniform color -- computed from all
    # categories' off-diagonal values up front, one shared scale for
    # honest cross-category comparison.
    all_matrices = {}
    for category in categories:
        row_idx = np.where(ref_categories == category)[0]
        all_matrices[category] = (row_idx, rsa_matrix_for_subset(aligned, model_names, row_idx))
    iu = np.triu_indices(len(model_names), k=1)
    all_offdiag = np.concatenate([m[iu] for _, m in all_matrices.values()])
    vmin, vmax = float(all_offdiag.min()), float(all_offdiag.max())

    im = None
    for ax, category in zip(axes, categories):
        row_idx, matrix = all_matrices[category]
        im = ax.imshow(matrix, vmin=vmin, vmax=vmax, cmap="viridis")
        ax.set_xticks(range(len(model_names)))
        ax.set_yticks(range(len(model_names)))
        ax.set_xticklabels(model_names, rotation=90, fontsize=6)
        ax.set_yticklabels(model_names, fontsize=6)
        ax.set_title(f"{category} (n={len(row_idx)})", fontsize=11, pad=10)
        print(f"  {category}: n={len(row_idx)}, mean off-diag RSA={matrix[iu].mean():.3f}")

    for ax in axes[len(categories):]:
        ax.axis("off")

    fig.suptitle("Model-vs-model RSA per probe-set category (final-layer embeddings, 18 models)", fontsize=14)
    fig.subplots_adjust(hspace=0.6, wspace=0.35, top=0.92)
    fig.colorbar(im, ax=axes.tolist(), fraction=0.02, pad=0.02, label=f"RSA (Spearman), range {vmin:.2f}-{vmax:.2f}")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"wrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "probe_set_manifest.csv")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "results" / "rsa_by_category.png")
    args = parser.parse_args()
    main(args.embeddings_dir, args.manifest, args.out)
