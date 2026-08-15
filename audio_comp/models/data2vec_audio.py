"""Data2Vec-audio — self-distillation-with-EMA-teacher (data2vec-family,
same lineage as music2vec but for the SPEECH domain instead of music).
Stage 2 addition (2026-08-12), part of the push toward ~20 models -- also a
genuine gap-filler: music2vec was previously this project's only
data2vec-family data point (music domain only); this adds a same-paradigm,
different-domain comparison. Verified against the HF model card: this is
`facebook/data2vec-audio-base`, the pure self-supervised checkpoint, NOT
`facebook/data2vec-audio-base-960h` (which is ASR-fine-tuned) -- checked
directly since this project's own base/self-supervised-only policy makes
that distinction load-bearing. License: Apache-2.0.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import Data2VecAudioModel, Wav2Vec2FeatureExtractor

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample


@register_model("data2vec_audio")
class Data2VecAudioEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="data2vec_audio",
        hf_id="facebook/data2vec-audio-base",
        paradigm="data2vec-family (self-distillation, EMA-updated teacher; speech)",
        license="Apache-2.0",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.info.hf_id)
        self._model = Data2VecAudioModel.from_pretrained(self.info.hf_id).to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        inputs = self._extractor(
            resampled, sampling_rate=self.info.expected_sample_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return mean_pool(outputs.last_hidden_state).cpu().numpy()
