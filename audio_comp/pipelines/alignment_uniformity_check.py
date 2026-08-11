"""Stage 6 diagnostic candidate: compute Wang & Isola alignment/uniformity
(audio_comp/geometry/alignment_uniformity.py) for every model on the
probe set, then check whether they predict downstream KNN scores (the
"raw embedding-space separability" side of the 2026-08-10 three-way
separability split) better than MLP scores (the "trainable-head-
exploitability" side) -- replacing the earlier silhouette-score proxy
with the real metric the theory actually motivates.

Two correlation scopes, both using only already-collected data (no new
extraction/Slurm jobs):
  - BirdCLEF (all 9 models) -- the only task every current model has run.
  - FMA-genre + UrbanSound8K average (7 models: the original roster,
    pre-`panns_cnn14`/`bird_mae`) -- matches the exact task pairing the
    original silhouette-vs-MLP correlation used (2026-08-10 earlier
    entry), so this is a direct, comparable upgrade of that check with
    the real metric instead of the proxy.

Usage:
    python -m audio_comp.pipelines.alignment_uniformity_check \
        --embeddings-dir $SCRATCH/audio_comp/embeddings \
        --xares-results results/xares_results.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

from audio_comp.geometry.alignment_uniformity import alignment_score, uniformity_score

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_categories(manifest_path: Path) -> dict[str, str]:
    with open(manifest_path) as f:
        return {row["clip_id"]: row["category"] for row in csv.DictReader(f)}


def load_xares_scores(xares_csv: Path) -> list[dict]:
    with open(xares_csv) as f:
        return list(csv.DictReader(f))


def aggregate_scores(rows: list[dict], task_names: list[str]) -> dict[str, dict[str, float]]:
    """model -> {mlp: avg, knn: avg} across the given task names, only for
    models that have every requested task (avoids silently averaging
    over an uneven task set per model)."""
    by_model: dict[str, dict[str, list[float]]] = {}
    for row in rows:
        if row["task"] not in task_names:
            continue
        m = row["model"]
        by_model.setdefault(m, {"mlp": [], "knn": []})
        by_model[m]["mlp"].append(float(row["mlp_score"]))
        by_model[m]["knn"].append(float(row["knn_score"]))
    return {
        m: {"mlp": float(np.mean(v["mlp"])), "knn": float(np.mean(v["knn"]))}
        for m, v in by_model.items()
        if len(v["mlp"]) == len(task_names)
    }


def main(embeddings_dir: str, xares_results_csv: str, manifest_csv: str) -> None:
    categories = load_categories(Path(manifest_csv))
    xares_rows = load_xares_scores(Path(xares_results_csv))

    birdclef_scores = aggregate_scores(xares_rows, ["BirdCLEF (50-species, mteb/birdclef25-mini)"])
    fma_us8k_scores = aggregate_scores(xares_rows, ["Free Music Archive Small", "UrbanSound 8k"])

    align_uniform: dict[str, dict[str, float]] = {}
    for npz_path in sorted(Path(embeddings_dir).glob("*.npz")):
        model = npz_path.stem
        data = np.load(npz_path, allow_pickle=True)
        clip_ids, embeddings = data["clip_ids"], data["embeddings"]
        labels = np.array([categories[str(cid)] for cid in clip_ids])
        align = alignment_score(embeddings, labels)
        uniform = uniformity_score(embeddings)
        align_uniform[model] = {"alignment": align, "uniformity": uniform}
        print(f"{model:15s} alignment={align:.4f}  uniformity={uniform:.4f}")

    print()
    for scope_name, scores in [("BirdCLEF (9 models)", birdclef_scores), ("FMA+UrbanSound8K avg (7 models)", fma_us8k_scores)]:
        models = [m for m in scores if m in align_uniform]
        if len(models) < 4:
            print(f"=== {scope_name}: too few overlapping models ({len(models)}), skipping ===")
            continue
        align_vals = [align_uniform[m]["alignment"] for m in models]
        uniform_vals = [align_uniform[m]["uniformity"] for m in models]
        mlp_vals = [scores[m]["mlp"] for m in models]
        knn_vals = [scores[m]["knn"] for m in models]

        print(f"=== {scope_name} (n={len(models)}: {sorted(models)}) ===")
        for metric_name, metric_vals in [("alignment", align_vals), ("uniformity", uniform_vals)]:
            for target_name, target_vals in [("MLP", mlp_vals), ("KNN", knn_vals)]:
                rho, p = spearmanr(metric_vals, target_vals)
                print(f"  spearman({metric_name}, {target_name}) = {rho:+.3f}  (p={p:.3f})")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", default="/scratch/pdoshi/audio_comp/embeddings")
    parser.add_argument("--xares-results", default=str(REPO_ROOT / "results" / "xares_results.csv"))
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "probe_set_manifest.csv"))
    args = parser.parse_args()
    main(args.embeddings_dir, args.xares_results, args.manifest)
