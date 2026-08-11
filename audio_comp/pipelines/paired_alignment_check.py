"""Recompute alignment using true instance-level positive pairs (see
audio_comp/geometry/alignment_uniformity.py's module docstring for why
the category-proxy version turned out degenerate: perfectly rank-
correlated with uniformity, Spearman -1.0, across all 9 models tested
2026-08-10). Checks whether the true-pair version actually decorrelates
from uniformity, and re-runs the same KNN/MLP correlation check with it.

Requires embeddings already extracted for
data/augmented_pair_extraction_manifest.csv (500 original + 500
pitch-shifted-augmented clips) into $SCRATCH/audio_comp/embeddings_augmented/
-- see build_augmented_probe_subset.py and
scripts/slurm/extract_embeddings_augmented.sbatch.

Usage:
    python -m audio_comp.pipelines.paired_alignment_check
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from audio_comp.geometry.alignment_uniformity import alignment_score_paired, uniformity_score
from audio_comp.pipelines.alignment_uniformity_check import aggregate_scores, load_xares_scores

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(embeddings_augmented_dir: str, embeddings_dir: str, xares_results_csv: str) -> None:
    xares_rows = load_xares_scores(Path(xares_results_csv))
    birdclef_scores = aggregate_scores(xares_rows, ["BirdCLEF (50-species, mteb/birdclef25-mini)"])
    fma_us8k_scores = aggregate_scores(xares_rows, ["Free Music Archive Small", "UrbanSound 8k"])

    paired_align: dict[str, float] = {}
    uniform: dict[str, float] = {}
    for npz_path in sorted(Path(embeddings_augmented_dir).glob("*.npz")):
        model = npz_path.stem
        data = np.load(npz_path, allow_pickle=True)
        clip_ids, embeddings = data["clip_ids"], data["embeddings"]
        clip_ids = [str(c) for c in clip_ids]

        idx_orig = {c.rsplit("_", 1)[0]: i for i, c in enumerate(clip_ids) if c.endswith("_orig")}
        idx_aug = {c.rsplit("_", 1)[0]: i for i, c in enumerate(clip_ids) if c.endswith("_aug")}
        pair_ids = sorted(set(idx_orig) & set(idx_aug), key=int)
        emb_a = embeddings[[idx_orig[p] for p in pair_ids]]
        emb_b = embeddings[[idx_aug[p] for p in pair_ids]]

        paired_align[model] = alignment_score_paired(emb_a, emb_b)
        # Uniformity from the same 500 original clips (not the augmented
        # views), for full internal consistency of this check.
        uniform[model] = uniformity_score(embeddings[[idx_orig[p] for p in pair_ids]], sample_size=500)
        print(f"{model:15s} paired_alignment={paired_align[model]:.4f}  uniformity(n=500)={uniform[model]:.4f}")

    models = sorted(paired_align)
    rho, p = spearmanr([paired_align[m] for m in models], [uniform[m] for m in models])
    print(f"\nspearman(paired_alignment, uniformity) across {len(models)} models = {rho:+.3f} (p={p:.3f})")
    print("(compare to the category-proxy version's -1.000, p=0.000 from the earlier check)\n")

    for scope_name, scores in [("BirdCLEF", birdclef_scores), ("FMA+UrbanSound8K avg", fma_us8k_scores)]:
        common = [m for m in models if m in scores]
        if len(common) < 4:
            print(f"=== {scope_name}: too few overlapping models ({len(common)}), skipping ===")
            continue
        align_vals = [paired_align[m] for m in common]
        mlp_vals = [scores[m]["mlp"] for m in common]
        knn_vals = [scores[m]["knn"] for m in common]
        print(f"=== {scope_name} (n={len(common)}: {sorted(common)}) ===")
        for target_name, target_vals in [("MLP", mlp_vals), ("KNN", knn_vals)]:
            rho, p = spearmanr(align_vals, target_vals)
            print(f"  spearman(paired_alignment, {target_name}) = {rho:+.3f}  (p={p:.3f})")
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-augmented-dir", default="/scratch/pdoshi/audio_comp/embeddings_augmented")
    parser.add_argument("--embeddings-dir", default="/scratch/pdoshi/audio_comp/embeddings")
    parser.add_argument("--xares-results", default=str(REPO_ROOT / "results" / "xares_results.csv"))
    args = parser.parse_args()
    main(args.embeddings_augmented_dir, args.embeddings_dir, args.xares_results)
