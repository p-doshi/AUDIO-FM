"""Whisper — encoder-only representation extracted from a full encoder-decoder
ASR model. Stage 2 Tier 2 addition (2026-08-12), part of the push toward ~20
models. License verified directly against the HF model card: Apache-2.0.

Uses `WhisperModel` (the plain encoder-decoder base class) and only runs
`.encoder(...)`, never `.generate()` or the decoder -- consistent with this
project's policy of comparing representation models, not task output. Note
this is trained with ASR supervision (paired audio-text), unlike this
project's other "self-supervised" entries -- closer in spirit to AST's
supervised-training data point than to wav2vec2/hubert's masked-modeling
ones, just with a different supervision signal (transcripts, not labels).
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import WhisperModel, WhisperFeatureExtractor

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample


@register_model("whisper")
class WhisperEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="whisper",
        hf_id="openai/whisper-base",
        paradigm="encoder-decoder ASR (supervised, transcript-based)",
        license="Apache-2.0",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        self._extractor = WhisperFeatureExtractor.from_pretrained(self.info.hf_id)
        self._model = WhisperModel.from_pretrained(self.info.hf_id).to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        inputs = self._extractor(resampled, sampling_rate=self.info.expected_sample_rate, return_tensors="pt")
        input_features = inputs["input_features"].to(self.device)
        with torch.no_grad():
            encoder_out = self._model.encoder(input_features)
        return mean_pool(encoder_out.last_hidden_state).cpu().numpy()
