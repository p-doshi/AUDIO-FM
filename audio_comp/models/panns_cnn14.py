"""PANNs Cnn14 — pure-CNN, supervised (AudioSet-tagging), general-audio
model. Fills the "pure CNN architecture" gap in CLAUDE.md's Stage 2 plan:
every other model in this roster is transformer-based, so this is the
only ResNet/CNN-family inductive-bias data point.

Checkpoint provenance verified 2026-08-10 directly against the primary
source (github.com/qiuqiangkong/audioset_tagging_cnn), not a secondary
mirror: MIT-licensed (LICENSE.MIT), official checkpoint hosted by the
same author at https://zenodo.org/record/3987831. Run
`scripts/setup_panns.sh` once before using this adapter.

Uses the `panns_inference` pip package (same author, qiuqiangkong) only
for its `Cnn14` architecture class -- not its `AudioTagging` wrapper,
which hardcodes a $HOME checkpoint path and forces `DataParallel`
wrapping, neither of which fits this project's checkpoint-caching or
device-handling conventions. Cnn14 takes a raw waveform directly (its
own spectrogram/logmel extraction happens inside forward()) and returns
a dict with a 2048-d 'embedding' (penultimate layer, standard PANNs
convention) alongside the 527-class AudioSet tagging output; only the
embedding is used here.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import resample

EXTERNAL_DIR = Path(os.environ.get("AUDIO_COMP_EXTERNAL", os.path.expanduser("~/audio_comp_external")))
CHECKPOINT_PATH = EXTERNAL_DIR / "panns" / "Cnn14_mAP=0.431.pth"


@register_model("panns_cnn14")
class PANNsCnn14Encoder(BaseAudioEncoder):
    info = ModelInfo(
        name="panns_cnn14",
        hf_id="qiuqiangkong/audioset_tagging_cnn (Zenodo checkpoint, not HF-native)",
        paradigm="supervised, pure-CNN (AudioSet tagging)",
        license="MIT",
        expected_sample_rate=32000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        if not CHECKPOINT_PATH.exists():
            raise RuntimeError(
                f"PANNs Cnn14 checkpoint not found at {CHECKPOINT_PATH}. Run "
                "`scripts/setup_panns.sh` first."
            )
        from panns_inference.models import Cnn14  # pip dep: panns_inference, torchlibrosa

        self._model = Cnn14(
            sample_rate=self.info.expected_sample_rate,
            window_size=1024,
            hop_size=320,
            mel_bins=64,
            fmin=50,
            fmax=14000,
            classes_num=527,
        )
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu")
        self._model.load_state_dict(checkpoint["model"])
        self._model = self._model.to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        max_len = max(len(w) for w in resampled)
        batch = np.stack([np.pad(w, (0, max_len - len(w))) for w in resampled]).astype(np.float32)
        wav = torch.from_numpy(batch).to(self.device)
        with torch.no_grad():
            output = self._model(wav)
        return output["embedding"].cpu().numpy()
