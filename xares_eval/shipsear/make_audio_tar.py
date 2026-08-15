"""Extract the kept ShipsEar clips (3 classes, see build_manifest.py) from
the HF-hosted zip and package them as X-ARES "private" webdataset tars,
one per fold (= recording session, leave-one-session-out).

Clips are already 5s pre-segmented -- no re-chunking needed, unlike
DeepShip's make_audio_tar.py. Extracted wavs go to $SCRATCH (not $HOME),
matching this project's established convention for derived/bursty
small-file audio (see deepship/make_audio_tar.py's docstring).

Usage (after build_manifest.py has written data/shipsear_manifest.csv):
    python -m xares_eval.shipsear.make_audio_tar
"""
from __future__ import annotations

import csv
import os
import zipfile
from collections import defaultdict
from pathlib import Path

from huggingface_hub import hf_hub_download
from loguru import logger
from xares.audiowebdataset import write_audio_tar

SCRATCH_ENV_ROOT = os.environ.get("XARES_ENV_ROOT", "/scratch/pdoshi/audio_comp/xares_env/env")
CLIPS_DIR = Path(os.environ.get("SHIPSEAR_CLIPS_DIR", "/scratch/pdoshi/audio_comp/shipsear_clips"))


def _read_manifest(manifest_csv: str) -> list[dict]:
    with open(manifest_csv) as f:
        return list(csv.DictReader(f))


def _extract_clips(rows: list[dict], clips_dir: Path) -> dict[str, Path]:
    """Extract only the kept zip members to individual wav files, skipping
    ones already on disk. Returns zip_member -> local path."""
    clips_dir.mkdir(parents=True, exist_ok=True)
    needed = {r["zip_member"] for r in rows}
    out_paths = {m: clips_dir / Path(m).name for m in needed}
    missing = {m: p for m, p in out_paths.items() if not p.exists()}
    if missing:
        zip_path = hf_hub_download("peng7554/DS3500", "ShipsEar.zip", repo_type="dataset")
        with zipfile.ZipFile(zip_path) as z:
            for member, out_path in missing.items():
                with z.open(member) as src, open(out_path, "wb") as dst:
                    dst.write(src.read())
    return out_paths


def make_audio_tar(manifest_csv: str = "data/shipsear_manifest.csv", num_shards_per_fold: int = 1) -> None:
    rows = _read_manifest(manifest_csv)
    local_paths = _extract_clips(rows, CLIPS_DIR)

    fold_paths: dict[int, list[Path]] = defaultdict(list)
    fold_labels: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        fold = int(row["fold"])
        fold_paths[fold].append(local_paths[row["zip_member"]])
        fold_labels[fold].append({"class_name": row["class_name"], "label": int(row["label"])})

    env_dir = Path(SCRATCH_ENV_ROOT) / "shipsear"
    env_dir.mkdir(parents=True, exist_ok=True)

    for fold in sorted(fold_paths):
        paths = [str(p) for p in fold_paths[fold]]
        labels = fold_labels[fold]
        logger.info(f"fold {fold}: {len(paths)} clips")
        tar_path = env_dir / f"shipsear_fold{fold}_*.tar"
        write_audio_tar(
            audio_paths=paths,
            labels=labels,
            tar_path=str(tar_path),
            num_shards=num_shards_per_fold,
        )

    (env_dir / ".audio_tar_ready").touch()
    logger.info(f"ShipsEar tars ready under {env_dir}")


if __name__ == "__main__":
    make_audio_tar()
