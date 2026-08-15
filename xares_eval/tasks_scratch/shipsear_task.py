"""ShipsEar vessel-type task (custom, not upstream X-ARES) -- 3-way
classification (A/B/D, see xares_eval/shipsear/build_manifest.py for why
only 3 of the dataset's 5 classes) on the REAL (non-synthetic) recordings
from peng7554/DS3500's ShipsEar.zip.

Explicitly a bare-minimum check, not a research-grade one (per the user,
2026-08-13): 10 leave-one-session-out folds total across 3 classes, some
sessions are small (e.g. session '1_1' has only 26 clips) -- read any
resulting accuracy with that in mind. The confidential Stage 5 v2 data is
the actual research-grade real-world generalization test; this exists to
give a *some* signal for "is the representation geometrically-agreeing
models are actually right," per-model, on real (if limited) recordings,
cheaply and without the confidential-data password/streaming boundary.

`private=True` skips X-ARES's own Zenodo download path -- the tars must
already exist under env_root/shipsear/ before this task runs; run
`python -m xares_eval.shipsear.build_manifest` then
`python -m xares_eval.shipsear.make_audio_tar` first.
"""
from __future__ import annotations

import os

from xares.task import TaskConfig

SCRATCH_ENV_ROOT = os.environ.get("XARES_ENV_ROOT", "/scratch/pdoshi/audio_comp/xares_env/env")

NUM_FOLDS = 10  # one per recording session -- see build_manifest.py


def shipsear_config(encoder) -> TaskConfig:
    config = TaskConfig(
        encoder=encoder,
        env_root=SCRATCH_ENV_ROOT,
        eval_weight=1156,  # total kept clips across classes 0/1/3 (369+301+486)
        formal_name="ShipsEar (vessel type A/B/D, real recordings only)",
        k_fold_splits=list(range(NUM_FOLDS)),
        label_processor=lambda x: x["label"],
        name="shipsear",
        output_dim=3,
        private=True,
    )
    config.audio_tar_name_of_split = {fold: f"shipsear_fold{fold}_*.tar" for fold in config.k_fold_splits}
    config.encoded_tar_name_of_split = {
        fold: f"shipsear-wds-encoded-fold-{fold}-*.tar" for fold in config.k_fold_splits
    }
    return config
