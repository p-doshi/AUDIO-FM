"""Segment DeepShip recordings into fixed-length clips and package them as
X-ARES "private" webdataset tars (private=True in deepship_task.py skips
X-ARES's own Zenodo download path entirely -- see xares/task.py's
download_audio_tar/make_encoded_tar: a private task only needs the tar
files + the audio-ready marker present under env_root already).

10s non-overlapping clips (CLIP_LENGTH_S in build_manifest.py), matching
the crop_length=10 convention this project already uses for FMA. The
remainder at the end of each recording (< 10s) is dropped, not padded --
consistent with how n_clips is computed in build_manifest.py.

Clips are written to $SCRATCH (not $HOME) both as a matter of quota and
because $HOME has shown intermittent Lustre I/O errors this project
(2026-08-10 journal entries) -- writing ~1000 small wav files is exactly
the kind of bursty small-file I/O that tripped it before.

Usage (after build_manifest.py has written data/deepship_manifest.csv):
    python -m xares_eval.deepship.make_audio_tar
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import soundfile as sf
from loguru import logger
from xares.audiowebdataset import write_audio_tar

from xares_eval.deepship.build_manifest import CLIP_LENGTH_S, NUM_FOLDS

SCRATCH_ENV_ROOT = os.environ.get("XARES_ENV_ROOT", "/scratch/pdoshi/audio_comp/xares_env/env")
CLIPS_DIR = Path(os.environ.get("DEEPSHIP_CLIPS_DIR", "/scratch/pdoshi/audio_comp/deepship_clips"))


def _read_manifest(manifest_csv: str) -> list[dict]:
    with open(manifest_csv) as f:
        return list(csv.DictReader(f))


def _segment_recording(row: dict, clips_dir: Path) -> list[Path]:
    """Slice one recording into CLIP_LENGTH_S-second wav clips, skipping
    ones already on disk. Returns the ordered list of clip paths."""
    n_clips = int(row["n_clips"])
    class_dir = clips_dir / row["vessel_class"]
    class_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = [class_dir / f"{row['record_id']}_{i:03d}.wav" for i in range(n_clips)]
    if all(p.exists() for p in clip_paths):
        return clip_paths

    audio, sr = sf.read(row["file"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    clip_len_samples = int(CLIP_LENGTH_S * sr)
    for i, clip_path in enumerate(clip_paths):
        if clip_path.exists():
            continue
        start = i * clip_len_samples
        chunk = audio[start : start + clip_len_samples]
        sf.write(clip_path, chunk, sr)
    return clip_paths


def make_audio_tar(manifest_csv: str = "data/deepship_manifest.csv", num_shards_per_fold: int = 1) -> None:
    rows = _read_manifest(manifest_csv)

    fold_paths: dict[int, list[Path]] = defaultdict(list)
    fold_labels: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        clip_paths = _segment_recording(row, CLIPS_DIR)
        fold = int(row["fold"])
        for p in clip_paths:
            fold_paths[fold].append(p)
            fold_labels[fold].append({"vessel_class": row["vessel_class"]})

    env_dir = Path(SCRATCH_ENV_ROOT) / "deepship"
    env_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(NUM_FOLDS):
        paths = [str(p) for p in fold_paths[fold]]
        labels = fold_labels[fold]
        logger.info(f"fold {fold}: {len(paths)} clips")
        tar_path = env_dir / f"deepship_fold{fold}_*.tar"
        write_audio_tar(
            audio_paths=paths,
            labels=labels,
            tar_path=str(tar_path),
            num_shards=num_shards_per_fold,
        )

    (env_dir / ".audio_tar_ready").touch()
    logger.info(f"DeepShip tars ready under {env_dir}")


if __name__ == "__main__":
    make_audio_tar()
