"""Wraps xares_eval.tasks.freemusicarchive_genre_task to redirect
env_root to $SCRATCH. X-ARES's own default ("./env", CWD-relative) would
put ~8GB of downloaded data under $HOME, which doesn't have room (see the
2026-08-09 journal entry on the HF-cache quota issue — same class of
problem). env_root can only be set via the TaskConfig constructor, not an
environment variable, and the real task config lives in an external,
unmodified repo clone — hence this thin wrapper rather than editing it in
place.
"""
from __future__ import annotations

import os

from xares_eval.tasks.freemusicarchive_genre_task import fma_genre_config as _upstream_task_fn

SCRATCH_ENV_ROOT = os.environ.get("XARES_ENV_ROOT", "/scratch/pdoshi/audio_comp/xares_env/env")


def fma_genre_config(encoder):
    config = _upstream_task_fn(encoder)
    config.env_root = SCRATCH_ENV_ROOT
    return config
