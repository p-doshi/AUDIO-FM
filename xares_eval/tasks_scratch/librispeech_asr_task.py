"""Wraps xares_eval.tasks.librispeech_asr_task to redirect env_root to
$SCRATCH. See freemusicarchive_genre_task.py (this directory) for the
full rationale."""
from __future__ import annotations

import os

from xares_eval.tasks.librispeech_asr_task import librispeech_asr_config as _upstream_task_fn

SCRATCH_ENV_ROOT = os.environ.get("XARES_ENV_ROOT", "/scratch/pdoshi/audio_comp/xares_env/env")


def librispeech_asr_config(encoder):
    config = _upstream_task_fn(encoder)
    config.env_root = SCRATCH_ENV_ROOT
    return config
