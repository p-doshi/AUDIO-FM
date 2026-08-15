"""wav2vec2-Conformer — wav2vec2's masked-prediction objective with a
Conformer (convolution-augmented transformer) encoder instead of a plain
transformer, the closest within-roster test of architecture (not
paradigm/domain) as a geometry-shaping axis alongside PANNs' pure-CNN
comparison (CLAUDE.md Stage 3 interim finding: architecture family was not
supported as an independent axis there -- this is a second, more targeted
test, holding paradigm AND training data fixed to plain wav2vec2's own
960h LibriSpeech, varying only the encoder block design). Stage 2 addition
(2026-08-12), part of the push toward ~20 models. License verified
directly against the HF model card: Apache-2.0. Self-supervised base
checkpoint (`rel-pos-large`), not fine-tuned.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import Wav2Vec2ConformerModel, Wav2Vec2FeatureExtractor

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample


@register_model("wav2vec2_conformer")
class Wav2Vec2ConformerEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="wav2vec2_conformer",
        hf_id="facebook/wav2vec2-conformer-rel-pos-large",
        paradigm="masked modeling (speech, Conformer encoder architecture)",
        license="Apache-2.0",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.info.hf_id)
        self._model = Wav2Vec2ConformerModel.from_pretrained(self.info.hf_id).to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        inputs = self._extractor(
            resampled, sampling_rate=self.info.expected_sample_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return mean_pool(outputs.last_hidden_state).cpu().numpy()
