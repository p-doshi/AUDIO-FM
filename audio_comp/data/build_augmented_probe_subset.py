"""Build a stratified subset of the probe set plus a pitch-shifted
"augmented view" of each clip, for a true instance-level positive-pair
alignment metric (Wang & Isola) -- replacing the same-category proxy
that turned out to be perfectly rank-correlated with uniformity
(2026-08-10 journal entry: same-category pairs conflate "close because
genuinely similar" with "close because same broad label").

Pitch-shift (librosa.effects.pitch_shift, small +/-2 semitone range) is
a standard augmented-positive-pair choice in the contrastive audio
learning literature (SimCLR-style audio augmentation sets typically
include pitch shift, time stretch, noise addition) and, unlike
SpecAugment-style masking, operates on raw waveform -- directly
compatible with every model adapter's existing embed_batch(waveform,
sample_rate) interface with zero model-specific code.

100 clips/category x 5 categories = 500 original clips, 500 augmented
counterparts -- enough for a real pooled alignment estimate (500 true
positive pairs) while keeping the subsequent 9-model re-extraction fast.
Same probe-set categories, sampled deterministically (fixed seed) from
data/probe_set_manifest.csv.

Usage:
    python -m audio_comp.data.build_augmented_probe_subset
"""
from __future__ import annotations

import csv
import os
import random
from pathlib import Path

import librosa
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parents[2]

N_PER_CATEGORY = 100
PITCH_SHIFT_SEMITONES = 2.0
SEED = 0


def main(
    manifest_csv: str,
    data_root: str,
    out_dir: str,
    out_manifest_csv: str,
) -> None:
    with open(manifest_csv) as f:
        rows = list(csv.DictReader(f))

    by_category: dict[str, list[dict]] = {}
    for row in rows:
        by_category.setdefault(row["category"], []).append(row)

    rng = random.Random(SEED)
    sampled = []
    for category, category_rows in sorted(by_category.items()):
        sampled.extend(rng.sample(category_rows, min(N_PER_CATEGORY, len(category_rows))))

    out_dir_path = Path(out_dir)
    (out_dir_path / "original").mkdir(parents=True, exist_ok=True)
    (out_dir_path / "augmented").mkdir(parents=True, exist_ok=True)

    out_rows = []
    for i, row in enumerate(sampled):
        src_path = Path(data_root) / row["path"]
        audio, sr = librosa.load(src_path, sr=None, mono=True)

        orig_out = out_dir_path / "original" / f"{i:04d}.wav"
        sf.write(orig_out, audio, sr)

        shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=PITCH_SHIFT_SEMITONES)
        aug_out = out_dir_path / "augmented" / f"{i:04d}.wav"
        sf.write(aug_out, shifted, sr)

        out_rows.append(
            dict(
                pair_id=i,
                clip_id=row["clip_id"],
                category=row["category"],
                original_path=f"original/{i:04d}.wav",
                augmented_path=f"augmented/{i:04d}.wav",
            )
        )

    os.makedirs(os.path.dirname(out_manifest_csv) or ".", exist_ok=True)
    with open(out_manifest_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "clip_id", "category", "original_path", "augmented_path"])
        writer.writeheader()
        writer.writerows(out_rows)

    print(f"Wrote {len(out_rows)} original+augmented pairs -> {out_dir_path}, manifest -> {out_manifest_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default=str(REPO_ROOT / "data" / "probe_set_manifest.csv"))
    parser.add_argument(
        "--data-root",
        default=os.environ.get("AUDIO_COMP_DATA_ROOT", os.path.expanduser("~/audio_comp_data")),
    )
    parser.add_argument(
        "--out-dir",
        default="/scratch/pdoshi/audio_comp/augmented_probe_subset",
    )
    parser.add_argument(
        "--out-manifest",
        default=str(REPO_ROOT / "data" / "augmented_probe_subset_manifest.csv"),
    )
    args = parser.parse_args()
    main(args.manifest, args.data_root, args.out_dir, args.out_manifest)
