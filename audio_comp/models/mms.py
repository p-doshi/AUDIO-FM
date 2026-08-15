"""MMS (Massively Multilingual Speech) — wav2vec2-architecture checkpoint
pretrained with wav2vec2's own self-supervised objective on ~500,000 hours
of speech across 1400+ languages. Stage 2 addition (2026-08-12), part of the
push toward ~20 models -- also a genuine data-breadth data point: this
project's own breadth-hypothesis finding (CLAUDE.md Stage 1(b)) treats
training-distribution breadth as a real geometry-shaping axis, and MMS is
the most extreme breadth case in the roster by a wide margin (1400+
languages vs. the next broadest entries). Same underlying architecture
class as wav2vec2/hubert (`Wav2Vec2Model`), verified to load correctly
against this checkpoint specifically, not assumed from the class name
alone. License verified directly against the HF model card: CC-BY-NC-4.0
(same tier as mert/music2vec, non-commercial research use unrestricted).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import Wav2Vec2FeatureExtractor, Wav2Vec2Model

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample


@register_model("mms")
class MMSEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="mms",
        hf_id="facebook/mms-300m",
        paradigm="masked modeling (speech, extreme training-distribution breadth: 1400+ languages)",
        license="CC-BY-NC-4.0",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
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
