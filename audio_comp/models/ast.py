"""AST (Audio Spectrogram Transformer) — supervised, transformer-based,
general-audio (AudioSet tagging) model. Fills the "supervised training"
gap in CLAUDE.md's Stage 2 plan, and specifically isolates supervision
from architecture: PANNs (also supervised, AudioSet) is a pure CNN, AST
is a transformer, so together they test whether the CNN-vs-transformer
axis (already found not to explain RSA disagreement, 2026-08-10) and the
supervised-vs-self-supervised axis are actually separable rather than
confounded in the roster.

Checkpoint provenance verified 2026-08-10 against the primary source
(github.com/YuanGongND/ast, the paper's own repo, Interspeech 2021):
BSD-3-Clause, matches the license stated on `MIT/ast-finetuned-audioset-
10-10-0.4593`'s HF card exactly — "MIT" here is the org name (the
authors' institution), not a license name; the actual license is BSD-3.

Uses the AudioSet-trained checkpoint, not a "base, non-finetuned"
variant, unlike wav2vec2/hubert elsewhere in this project. That's not a
violation of this project's "avoid -ft checkpoints" convention — it's
the correct application of the underlying principle. wav2vec2/hubert
have a genuine two-stage structure (self-supervised pretrain, then an
optional ASR-specific finetune on top) where the finetuned stage would
conflate general representation quality with task-specific tuning; AST
has no such two-stage structure to begin with. Its supervised AudioSet
training *is* its designed pretraining — there is no "non-finetuned AST"
to prefer instead, and CLAUDE.md's Tier 1 table lists AST specifically
to test what supervised training (as the sole training signal) does to
representational geometry, which requires exactly this checkpoint.

Uses `pooler_output` (the model's own designed clip-level pooling head)
rather than manually mean-pooling `last_hidden_state`, same reasoning as
CLAP's wrapper: prefer the representation the model was actually built
to produce over an arbitrary alternative.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTModel

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import resample


@register_model("ast")
class ASTEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="ast",
        hf_id="MIT/ast-finetuned-audioset-10-10-0.4593",
        paradigm="supervised, transformer (AudioSet tagging)",
        license="BSD-3-Clause",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        self._extractor = ASTFeatureExtractor.from_pretrained(self.info.hf_id)
        self._model = ASTModel.from_pretrained(self.info.hf_id).to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        inputs = self._extractor(resampled, sampling_rate=self.info.expected_sample_rate, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return outputs.pooler_output.cpu().numpy()
