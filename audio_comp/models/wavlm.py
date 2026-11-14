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
from ._wavlm_lora_attention_fix import monkeypatch_wavlm_attention

# Real bug, root-caused 2026-09-11: transformers WavLMAttention.torch_multi_head_self_attention
# extracts q_proj/k_proj/v_proj .weight TENSORS directly and calls F.multi_head_attention_forward,
# never invoking q_proj.forward(x)/v_proj.forward(x) as normal submodule calls -- so any module
# wrapping (PEFT LoRA, this project's own ALLoRALinear) attached to those submodules is silently
# disconnected from the forward computation graph. Verified via direct gradient test: 48/48 LoRA
# params and 48/50 ALLoRA params got grad=None (0 params changed after an optimizer step) before
# this patch; 0/48 and 0/50 after. See _wavlm_lora_attention_fix.py's docstring for the full
# diagnosis and the project notes 2026-09-11. Applied at import time, idempotent, a pure refactor of *how*
# attention is computed (manual q/k/v/out_proj submodule calls instead of the fused functional op)
# -- verified numerically equivalent to stock in eval mode (max abs diff 3.6e-5, float noise only).
# Scoped to WavLM only via monkeypatch_wavlm_attention()'s own module-level target; harmless for
# frozen-embedding extraction (no adapter attached there, so the bug never manifested for that path).
monkeypatch_wavlm_attention()



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
