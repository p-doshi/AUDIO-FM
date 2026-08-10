"""BirdCLEF species-classification task (custom, not upstream X-ARES) --
50-way classification on `mteb/birdclef25-mini` (50 species x 20
recordings each), packaged by xares_eval/birdclef/make_audio_tar.py.

`private=True` skips X-ARES's own Zenodo download path (see
xares/task.py) -- tars must already exist under env_root/birdclef/
before this task runs: `python -m xares_eval.birdclef.build_manifest`
then `python -m xares_eval.birdclef.make_audio_tar` first.

k_fold_splits are recording-grouped (each of the 1000 source recordings
assigned to exactly one fold; all its clips travel with it) -- see
xares_eval/birdclef/build_manifest.py's module docstring. Unlike
DeepShip, this is a clean, unambiguous leakage-control split: every
recording's row->audio->label mapping is directly verified (each `url`
is unique across the dataset), no metadata-join uncertainty here.

CLASS_LABEL_MAPS is derived from the manifest CSV itself (sorted unique
species codes) rather than hardcoded, so it can't silently drift out of
sync with what make_audio_tar.py actually packaged.
"""
from __future__ import annotations

import csv
import os

from xares.task import TaskConfig

SCRATCH_ENV_ROOT = os.environ.get("XARES_ENV_ROOT", "/scratch/pdoshi/audio_comp/xares_env/env")
MANIFEST_CSV = os.environ.get(
    "BIRDCLEF_MANIFEST_CSV", os.path.join(os.path.dirname(__file__), "..", "..", "data", "birdclef_manifest.csv")
)

NUM_FOLDS = 5


def _load_class_label_maps() -> dict[str, int]:
    with open(MANIFEST_CSV) as f:
        species = sorted({row["species"] for row in csv.DictReader(f)})
    return {s: i for i, s in enumerate(species)}


CLASS_LABEL_MAPS = _load_class_label_maps()


def birdclef_config(encoder) -> TaskConfig:
    config = TaskConfig(
        encoder=encoder,
        env_root=SCRATCH_ENV_ROOT,
        eval_weight=1000,
        formal_name="BirdCLEF (50-species, mteb/birdclef25-mini)",
        k_fold_splits=list(range(NUM_FOLDS)),
        label_processor=lambda x: CLASS_LABEL_MAPS[x["species"]],
        name="birdclef",
        output_dim=len(CLASS_LABEL_MAPS),
        private=True,
    )
    config.audio_tar_name_of_split = {fold: f"birdclef_fold{fold}_*.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {
        fold: f"birdclef-wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits
    }
    return config
