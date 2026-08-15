"""SEW (Squeezed and Efficient Wav2vec, ASAPP) — a compute-efficiency-
focused masked-modeling speech model, architecturally distinct from the
plain-transformer wav2vec2/hubert family (squeezed context network,
different layer/stride design aimed at faster inference at comparable
quality). Stage 2 addition (2026-08-12), part of the push toward ~20
models. License verified directly against the HF model card: Apache-2.0.
Self-supervised base checkpoint (`sew-tiny-100k`), not fine-tuned.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import SEWModel, Wav2Vec2FeatureExtractor

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample


@register_model("sew")
class SEWEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="sew",
        hf_id="asapp/sew-tiny-100k",
        paradigm="masked modeling (speech, efficiency-focused architecture)",
        license="Apache-2.0",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.info.hf_id)
        self._model = SEWModel.from_pretrained(self.info.hf_id).to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        inputs = self._extractor(
            resampled, sampling_rate=self.info.expected_sample_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return mean_pool(outputs.last_hidden_state).cpu().numpy()
