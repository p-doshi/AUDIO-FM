"""MusicFM — masked-modeling (BEST-RQ) music model.

Unlike the other kickoff models this isn't natively `transformers`-loadable:
it needs the upstream github.com/minzwon/musicfm repo importable on
sys.path, plus a stats file and checkpoint downloaded from the model's HF
repo (`minzwon/MusicFM`). Run `scripts/setup_musicfm.sh` once before using
this adapter — `load()` raises a clear error if that hasn't happened yet.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample

EXTERNAL_DIR = Path(os.environ.get("AUDIO_COMP_EXTERNAL", os.path.expanduser("~/audio_comp_external")))
MUSICFM_REPO = EXTERNAL_DIR / "musicfm"
MUSICFM_DATA_DIR = MUSICFM_REPO / "data"


@register_model("musicfm")
class MusicFMEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="musicfm",
        hf_id="minzwon/MusicFM",
        paradigm="masked modeling (BEST-RQ, music)",
        license="MIT",
        expected_sample_rate=24000,
    )

    # Layer 7 of 12 is the upstream README's recommended general-purpose
    # embedding layer (layer 12 is meant for fine-tuning, not frozen probing).
    layer_ix: int = 7

    def load(self) -> None:
        if not MUSICFM_REPO.exists():
            raise RuntimeError(
                f"MusicFM repo not found at {MUSICFM_REPO}. Run "
                "`scripts/setup_musicfm.sh` first (clones minzwon/musicfm and "
                "downloads pretrained_msd.pt + msd_stats.json from the HF repo)."
            )
        if str(EXTERNAL_DIR) not in sys.path:
            sys.path.insert(0, str(EXTERNAL_DIR))
        from musicfm.model.musicfm_25hz import MusicFM25Hz  # external repo, not a package dep

        self._model = (
            MusicFM25Hz(
                is_flash=False,
                stat_path=str(MUSICFM_DATA_DIR / "msd_stats.json"),
                model_path=str(MUSICFM_DATA_DIR / "pretrained_msd.pt"),
            )
            .to(self.device)
            .eval()
        )

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        max_len = max(len(w) for w in resampled)
        batch = np.stack([np.pad(w, (0, max_len - len(w))) for w in resampled]).astype(np.float32)
        wav = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            emb = self._model.get_latent(wav, layer_ix=self.layer_ix)
        return mean_pool(emb).cpu().numpy()
