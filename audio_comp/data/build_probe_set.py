"""Build the fixed probe-set manifest from configs/categories.yaml.

Usage:
    python -m audio_comp.data.build_probe_set

Writes:
    - audio files under $AUDIO_COMP_DATA_ROOT/probe_set/<category>/<clip_id>.wav
    - data/probe_set_manifest.csv — the ONLY probe-set artifact tracked in
      git; re-derivable from this script + the category configs.

Scaling the probe set later (20 -> thousands/category) is a one-line change
to `n_per_category` in configs/categories.yaml, re-run with the same seed.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import soundfile as sf
import yaml

from audio_comp.data import get_dataset_class

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(os.environ.get("AUDIO_COMP_DATA_ROOT", os.path.expanduser("~/audio_comp_data")))


def build(config_path: Path, manifest_path: Path) -> None:
    with open(config_path) as f:
        config = yaml.safe_load(f)

    seed = config["seed"]
    default_n = config["n_per_category"]
    probe_set_dir = DATA_ROOT / "probe_set"
    probe_set_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for category, cat_config in config["categories"].items():
        source_name = cat_config["source"]
        n = cat_config.get("n_per_category", default_n)
        segment_seconds = cat_config.get("segment_seconds")

        source = get_dataset_class(source_name)()
        out_dir = probe_set_dir / category
        out_dir.mkdir(parents=True, exist_ok=True)

        count = 0
        for clip in source.iter_clips(seed=seed, segment_seconds=segment_seconds):
            if count >= n:
                break
            out_path = out_dir / (clip.clip_id.replace("/", "__") + ".wav")
            sf.write(out_path, clip.waveform, clip.sample_rate)
            rows.append(
                {
                    "clip_id": clip.clip_id,
                    "category": clip.category,
                    "source_dataset": clip.source_dataset,
                    "license": source.info.license,
                    "path": str(out_path.relative_to(DATA_ROOT)),
                    "sample_rate": clip.sample_rate,
                    "duration_sec": round(clip.duration_sec, 3),
                }
            )
            count += 1

        if count < n:
            raise RuntimeError(
                f"category '{category}': only got {count}/{n} clips from '{source_name}' "
                "(source exhausted, or too many candidates were filtered out)"
            )
        print(f"{category}: wrote {count} clips from {source_name}")

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"wrote manifest with {len(rows)} clips to {manifest_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "configs" / "categories.yaml")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "probe_set_manifest.csv")
    args = parser.parse_args()
    build(args.config, args.manifest)
