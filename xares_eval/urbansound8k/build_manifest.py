"""Build a local-file manifest for UrbanSound8K, for the LoRA-vs-frozen
matched fine-tuning comparison (not X-ARES's own Zenodo-tar-hosted eval
pipeline, which this project's own raw-audio scripts can't fine-tune
against directly). Mirrors xares_eval/birdclef/build_manifest.py's
pattern: decode each HF row to a local .wav once, write a CSV manifest.

Uses UrbanSound8K's own native `fold` field (1-10) -- the dataset's own
standard, widely-used 10-fold CV split (Salamon et al. 2014's original
protocol), not a custom split -- for the "each dataset's own natural
held-out split" comparison design.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import soundfile as sf

HF_DATASET = "danavery/urbansound8K"
OUT_MANIFEST = Path(os.environ.get("US8K_MANIFEST_CSV", "data/urbansound8k_manifest.csv"))
AUDIO_OUT_DIR = Path(os.environ.get("US8K_AUDIO_DIR", "/scratch/pdoshi/audio_comp/urbansound8k_audio"))


def build_rows(audio_out_dir: Path) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for i in range(len(ds)):
        row = ds[i]
        wav_path = audio_out_dir / f"{i:05d}_{row['class']}.wav"
        if not wav_path.exists():
            audio = row["audio"]
            samples = audio["array"] if isinstance(audio, dict) else audio.get_all_samples().data.numpy()
            sr = audio["sampling_rate"] if isinstance(audio, dict) else audio.get_all_samples().sample_rate
            if hasattr(samples, "ndim") and samples.ndim > 1:
                samples = samples.mean(axis=0)
            sf.write(wav_path, samples, sr)

        rows.append(
            dict(
                file=str(wav_path),
                row_index=i,
                slice_file_name=row["slice_file_name"],
                fsID=row["fsID"],
                classID=row["classID"],
                soundevent=row["class"],
                fold=row["fold"],
            )
        )
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{len(ds)} processed", flush=True)
    return rows


def main() -> None:
    rows = build_rows(AUDIO_OUT_DIR)
    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["file", "row_index", "slice_file_name", "fsID", "classID", "soundevent", "fold"]
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} clips -> {OUT_MANIFEST}, folds 1-10 (UrbanSound8K's own standard split)")


if __name__ == "__main__":
    main()
