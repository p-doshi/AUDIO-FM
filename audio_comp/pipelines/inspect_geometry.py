"""Sanity-check one model's own RDM against category labels, independent of
cross-model RSA/CKA agreement.

A model can fail to agree with every other teacher for two very different
reasons: (a) it organizes the probe set by genuinely idiosyncratic-but-real
structure (an isolated-geometry finding, scientifically interesting), or
(b) its embedding space has collapsed (e.g. everything roughly equidistant)
and "doesn't correlate with anyone" is really "doesn't encode anything."
RSA/CKA against other models can't distinguish these — this script checks
the model's RDM against the probe set's own category labels instead.

Usage:
    python -m audio_comp.pipelines.inspect_geometry --model music2vec \
        --embeddings-dir $SCRATCH/audio_comp/embeddings

Reports:
    - mean/std pairwise distance within-category vs. between-category
    - a separation index: (between_mean - within_mean) / pooled_std
      (near 0 => categories aren't separated; the higher the better)
    - coefficient of variation of ALL pairwise distances — the collapse
      signature is a *low* CV (everything roughly equidistant) regardless
      of whether categories separate
    - a compact category x category mean-distance matrix
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from audio_comp.geometry.rdm import compute_rdm

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_clip_categories(manifest_path: Path) -> dict:
    with open(manifest_path) as f:
        return {row["clip_id"]: row["category"] for row in csv.DictReader(f)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "probe_set_manifest.csv")
    parser.add_argument("--rdm-metric", default="correlation")
    args = parser.parse_args()

    data = np.load(args.embeddings_dir / f"{args.model}.npz", allow_pickle=True)
    clip_ids, embeddings = data["clip_ids"], data["embeddings"]
    clip_to_category = load_clip_categories(args.manifest)
    categories = np.array([clip_to_category[cid] for cid in clip_ids])

    rdm = compute_rdm(embeddings, metric=args.rdm_metric)
    n = rdm.shape[0]
    iu = np.triu_indices(n, k=1)
    distances = rdm[iu]
    same_category = categories[iu[0]] == categories[iu[1]]

    within = distances[same_category]
    between = distances[~same_category]
    pooled_std = np.sqrt((within.var() + between.var()) / 2)
    separation_index = (between.mean() - within.mean()) / pooled_std if pooled_std > 0 else float("nan")
    cv = distances.std() / distances.mean() if distances.mean() != 0 else float("nan")

    print(f"model: {args.model}  (n={n} clips, metric={args.rdm_metric})")
    print(f"within-category distance:  mean={within.mean():.4f}  std={within.std():.4f}")
    print(f"between-category distance: mean={between.mean():.4f}  std={between.std():.4f}")
    print(f"separation index (between-within)/pooled_std: {separation_index:.4f}")
    print(f"coefficient of variation of ALL pairwise distances: {cv:.4f}"
          f"  {'<-- LOW: possible collapse (everything ~equidistant)' if cv < 0.1 else ''}")

    cats = sorted(set(categories))
    print("\ncategory x category mean distance:")
    header = "".ljust(14) + "".join(c[:10].ljust(12) for c in cats)
    print(header)
    for ca in cats:
        row = []
        for cb in cats:
            mask = ((categories[iu[0]] == ca) & (categories[iu[1]] == cb)) | (
                (categories[iu[0]] == cb) & (categories[iu[1]] == ca)
            )
            row.append(f"{distances[mask].mean():.4f}".ljust(12))
        print(ca[:12].ljust(14) + "".join(row))


if __name__ == "__main__":
    main()
