"""Package MIMII clips (already local, already 10s pre-segmented -- no
extraction/segmentation step needed, simpler than both deepship's and
shipsear's make_audio_tar.py) into X-ARES "private" webdataset tars, one
per fold (= physical machine unit, leave-one-unit-out).

Usage (after build_manifest.py has written data/mimii_manifest.csv):
    python -m xares_eval.mimii.make_audio_tar
"""
from __future__ import annotations

import csv
import os
from collections import defaultdict
from pathlib import Path

from loguru import logger
from xares.audiowebdataset import write_audio_tar

SCRATCH_ENV_ROOT = os.environ.get("XARES_ENV_ROOT", "/scratch/pdoshi/audio_comp/xares_env/env")


def _read_manifest(manifest_csv: str) -> list[dict]:
    with open(manifest_csv) as f:
        return list(csv.DictReader(f))


def make_audio_tar(manifest_csv: str = "data/mimii_manifest.csv", num_shards_per_fold: int = 1) -> None:
    rows = _read_manifest(manifest_csv)

    fold_paths: dict[int, list[str]] = defaultdict(list)
    fold_labels: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        fold = int(row["fold"])
        fold_paths[fold].append(row["file"])
        fold_labels[fold].append({"condition": row["condition"], "label": int(row["label"])})

    env_dir = Path(SCRATCH_ENV_ROOT) / "mimii"
    env_dir.mkdir(parents=True, exist_ok=True)

    for fold in sorted(fold_paths):
        paths = fold_paths[fold]
        labels = fold_labels[fold]
        logger.info(f"fold {fold}: {len(paths)} clips")
        tar_path = env_dir / f"mimii_fold{fold}_*.tar"
        write_audio_tar(
            audio_paths=paths,
            labels=labels,
            tar_path=str(tar_path),
            num_shards=num_shards_per_fold,
        )

    (env_dir / ".audio_tar_ready").touch()
    logger.info(f"MIMII tars ready under {env_dir}")


if __name__ == "__main__":
    make_audio_tar()
