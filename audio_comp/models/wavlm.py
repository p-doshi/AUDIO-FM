"""WavLM — masked-modeling speech representation model (self-supervised
checkpoint, not an ASR/task-fine-tuned variant). Stage 2 Tier 2 addition
(2026-08-12): another speech masked-modeling variant alongside
hubert/wav2vec2, not a new representational axis -- lowest-engineering-risk
Tier 2 pick, added while the cluster GPU queue is congested and Tier 2 work
doesn't need to wait on it.

Checkpoint/license verified directly against microsoft/UniSpeech (the repo
the model's own HuggingFace card points to for licensing, not microsoft/unilm
-- a different repo than BEATs, not assumed to share BEATs' MIT license just
because both are Microsoft speech projects). UniSpeech's root LICENSE is
CC BY-SA 3.0 Unported (verified 2026-08-12 by reading the file directly) --
a license type not otherwise in this project's roster (not MIT like BEATs,
not CC-BY-NC like mert/music2vec). No separate statement for the model
weights specifically exists beyond the HF card's explicit link to this
license file; read as covering the checkpoint too by the same
absence-of-carve-out reasoning already used for BEATs, not a formality --
re-verify if this becomes a redistribution question rather than the
research/embedding-extraction use this project makes of it.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import WavLMModel, Wav2Vec2FeatureExtractor

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample


@register_model("wavlm")
class WavLMEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="wavlm",
        hf_id="microsoft/wavlm-base-plus",
        paradigm="masked modeling (speech)",
        license="CC BY-SA 3.0 Unported",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.info.hf_id)
        self._model = WavLMModel.from_pretrained(self.info.hf_id).to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        inputs = self._extractor(
            resampled, sampling_rate=self.info.expected_sample_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return mean_pool(outputs.last_hidden_state).cpu().numpy()
