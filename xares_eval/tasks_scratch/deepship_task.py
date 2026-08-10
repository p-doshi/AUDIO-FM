"""DeepShip vessel-class task (custom, not upstream X-ARES) -- 4-way
classification (cargo/passengership/tanker/tug) on the 63-clip GitHub
subset of DeepShip, packaged by xares_eval/deepship/make_audio_tar.py.

`private=True` skips X-ARES's own Zenodo download path entirely (see
xares/task.py's download_audio_tar/make_encoded_tar) -- the tars must
already exist under env_root/deepship/ before this task runs; run
`python -m xares_eval.deepship.build_manifest` then
`python -m xares_eval.deepship.make_audio_tar` first (see
scripts/setup_deepship.sh for fetching the raw data).

k_fold_splits are file-grouped, not vessel-grouped -- see
xares_eval/deepship/build_manifest.py's module docstring for why
(DeepShip's metafile record_id is not a reliable join key to the hosted
wav filenames, discovered 2026-08-10; vessel identity can't be trusted
for ~43% of the 63 files). This still blocks same-recording-pass leakage
across train/test, but can't guarantee the same physical vessel never
appears in two different files across folds. State this caveat wherever
DeepShip results are reported -- this task is a weaker leakage-control
guarantee than ESC-50/UrbanSound8K's official folds.
"""
from __future__ import annotations

import os

from xares.task import TaskConfig

SCRATCH_ENV_ROOT = os.environ.get("XARES_ENV_ROOT", "/scratch/pdoshi/audio_comp/xares_env/env")

CLASS_LABEL_MAPS = {
    "cargo": 0,
    "passengership": 1,
    "tanker": 2,
    "tug": 3,
}

NUM_FOLDS = 3


def deepship_config(encoder) -> TaskConfig:
    config = TaskConfig(
        encoder=encoder,
        env_root=SCRATCH_ENV_ROOT,
        eval_weight=63,
        formal_name="DeepShip (vessel class, 63-clip subset)",
        k_fold_splits=list(range(NUM_FOLDS)),
        label_processor=lambda x: CLASS_LABEL_MAPS[x["vessel_class"]],
        name="deepship",
        output_dim=len(CLASS_LABEL_MAPS),
        private=True,
    )
    config.audio_tar_name_of_split = {fold: f"deepship_fold{fold}_*.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {
        fold: f"deepship-wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits
    }
    return config
