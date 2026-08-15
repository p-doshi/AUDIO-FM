"""UniSpeech-SAT — masked-modeling speech representation model with an
added speaker-aware (SAT: Speaker Aware Training) pretraining objective on
top of the standard wav2vec2-style masked prediction. Stage 2 addition
(2026-08-12), part of the push toward ~20 models. Self-supervised base
checkpoint, not the downstream-fine-tuned variant.

Same repo/license situation as WavLM (`audio_comp/models/wavlm.py`) --
`microsoft/unispeech-sat-base`'s HF card points to `microsoft/UniSpeech`'s
root LICENSE, already verified 2026-08-12 for WavLM: CC BY-SA 3.0 Unported.
Not re-fetched here since it's the same repo, same file, same reasoning.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import UniSpeechSatModel, Wav2Vec2FeatureExtractor

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample


@register_model("unispeech_sat")
class UniSpeechSatEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="unispeech_sat",
        hf_id="microsoft/unispeech-sat-base",
        paradigm="masked modeling + speaker-aware pretraining (speech)",
        license="CC BY-SA 3.0 Unported",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.info.hf_id)
        self._model = UniSpeechSatModel.from_pretrained(self.info.hf_id).to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        inputs = self._extractor(
            resampled, sampling_rate=self.info.expected_sample_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return mean_pool(outputs.last_hidden_state).cpu().numpy()
