"""wav2vec 2.0 — masked-modeling speech representation model (self-supervised
checkpoint, not the ASR-fine-tuned variant)."""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample


@register_model("wav2vec2")
class Wav2Vec2Encoder(BaseAudioEncoder):
    info = ModelInfo(
        name="wav2vec2",
        hf_id="facebook/wav2vec2-large-lv60",
        paradigm="masked modeling (speech)",
        license="Apache-2.0",
        expected_sample_rate=16000,
    )

    def load(self) -> None:
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.info.hf_id)
        self._model = Wav2Vec2Model.from_pretrained(self.info.hf_id).to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        inputs = self._extractor(
            resampled, sampling_rate=self.info.expected_sample_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return mean_pool(outputs.last_hidden_state).cpu().numpy()
