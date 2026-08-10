"""Segment BirdCLEF recordings into 5s clips and package them as X-ARES
"private" webdataset tars. See build_manifest.py's module docstring for
the clip-length and fold-assignment rationale.

Recordings shorter than CLIP_LENGTH_S are zero-padded up to length
(unlike DeepShip, which drops remainders) -- see build_manifest.py for
why: dropping would remove whole recordings, and each species only has
20 to begin with.
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
from loguru import logger
from xares.audiowebdataset import write_audio_tar

from xares_eval.birdclef.build_manifest import CLIP_LENGTH_S, NUM_FOLDS

SCRATCH_ENV_ROOT = os.environ.get("XARES_ENV_ROOT", "/scratch/pdoshi/audio_comp/xares_env/env")
CLIPS_DIR = Path(os.environ.get("BIRDCLEF_CLIPS_DIR", "/scratch/pdoshi/audio_comp/birdclef_clips"))


def _read_manifest(manifest_csv: str) -> list[dict]:
    with open(manifest_csv) as f:
        return list(csv.DictReader(f))


def _segment_recording(row: dict, clips_dir: Path) -> list[Path]:
    n_clips = int(row["n_clips"])
    species_dir = clips_dir / row["species"]
    species_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = [species_dir / f"{row['row_index']}_{i:03d}.wav" for i in range(n_clips)]
    if all(p.exists() for p in clip_paths):
        return clip_paths

    audio, sr = sf.read(row["file"], dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    clip_len_samples = int(CLIP_LENGTH_S * sr)

    if len(audio) < clip_len_samples:
        # Single short recording: zero-pad up to CLIP_LENGTH_S.
        padded = np.zeros(clip_len_samples, dtype=audio.dtype)
        padded[: len(audio)] = audio
        sf.write(clip_paths[0], padded, sr)
        return clip_paths

    for i, clip_path in enumerate(clip_paths):
        if clip_path.exists():
            continue
        start = i * clip_len_samples
        chunk = audio[start : start + clip_len_samples]
        sf.write(clip_path, chunk, sr)
    return clip_paths


def make_audio_tar(manifest_csv: str = "data/birdclef_manifest.csv", num_shards_per_fold: int = 4) -> None:
    rows = _read_manifest(manifest_csv)

    fold_paths: dict[int, list[Path]] = defaultdict(list)
    fold_labels: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        clip_paths = _segment_recording(row, CLIPS_DIR)
        fold = int(row["fold"])
        for p in clip_paths:
            fold_paths[fold].append(p)
            fold_labels[fold].append({"species": row["species"]})

    env_dir = Path(SCRATCH_ENV_ROOT) / "birdclef"
    env_dir.mkdir(parents=True, exist_ok=True)

    for fold in range(NUM_FOLDS):
        paths = [str(p) for p in fold_paths[fold]]
        labels = fold_labels[fold]
        logger.info(f"fold {fold}: {len(paths)} clips")
        tar_path = env_dir / f"birdclef_fold{fold}_*.tar"
        write_audio_tar(
            audio_paths=paths,
            labels=labels,
            tar_path=str(tar_path),
            num_shards=num_shards_per_fold,
        )

    (env_dir / ".audio_tar_ready").touch()
    logger.info(f"BirdCLEF tars ready under {env_dir}")


if __name__ == "__main__":
    make_audio_tar()
