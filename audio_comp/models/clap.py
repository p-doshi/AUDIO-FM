"""LAION CLAP — contrastive audio-text model."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import ClapModel, ClapProcessor

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import resample


@register_model("clap")
class ClapEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="clap",
        hf_id="laion/larger_clap_general",
        paradigm="contrastive (audio-text)",
        license="Apache-2.0",
        expected_sample_rate=48000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        self._processor = ClapProcessor.from_pretrained(self.info.hf_id)
        self._model = ClapModel.from_pretrained(self.info.hf_id).to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        # `audios=` was renamed to `audio=` in newer transformers versions;
        # the old kwarg now raises instead of warning.
        inputs = self._processor(
            audio=resampled, sampling_rate=self.info.expected_sample_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            # newer transformers returns a BaseModelOutputWithPooling, not a
            # raw tensor; the projected+normalized embedding is pooler_output
            outputs = self._model.get_audio_features(**inputs)
        return outputs.pooler_output.cpu().numpy()
