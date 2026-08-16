"""MIMII industrial machine anomaly-detection task (custom, not upstream
X-ARES) -- binary classification (normal vs. abnormal machine sound),
the dataset's own canonical purpose. 16 leave-one-physical-unit-out
folds (4 machine types x 4 units each -- see
xares_eval/mimii/build_manifest.py for why unit-level grouping, not
clip-level).

Severe class imbalance, stated explicitly: 14,719 normal vs. 3,300
abnormal across the full 6dB tier (~4.5:1) -- read any accuracy number
with this in mind, a model that mostly predicts "normal" can still score
well above a naive 50% baseline without learning much about the
minority class.

`private=True` skips X-ARES's own Zenodo download path -- the tars must
already exist under env_root/mimii/ before this task runs; run
`python -m xares_eval.mimii.build_manifest` then
`python -m xares_eval.mimii.make_audio_tar` first.
"""
from __future__ import annotations

import os

from xares.task import TaskConfig

SCRATCH_ENV_ROOT = os.environ.get("XARES_ENV_ROOT", "/scratch/pdoshi/audio_comp/xares_env/env")

NUM_FOLDS = 16  # one per physical machine unit -- see build_manifest.py


def mimii_config(encoder) -> TaskConfig:
    config = TaskConfig(
        encoder=encoder,
        env_root=SCRATCH_ENV_ROOT,
        eval_weight=18019,
        formal_name="MIMII (industrial machine anomaly detection, normal vs. abnormal, 6dB tier)",
        k_fold_splits=list(range(NUM_FOLDS)),
        label_processor=lambda x: x["label"],
        name="mimii",
        output_dim=2,
        private=True,
    )
    config.audio_tar_name_of_split = {fold: f"mimii_fold{fold}_*.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {
        fold: f"mimii-wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits
    }
    return config
