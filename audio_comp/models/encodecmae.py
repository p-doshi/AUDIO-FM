"""EncodecMAE (Pepino et al. 2023, arXiv 2309.07391) — masked autoencoder
pretrained to predict discrete EnCodec-derived targets, filling this
project's neural-codec-derived representation gap (CLAUDE.md Stage 2
Tier 1's last remaining item) -- a genuinely different representation
type from every other model in the roster: predicts discrete tokens from
a *learned neural audio codec* (EnCodec), not raw spectrogram
reconstruction (audiomae/bird_mae) or a hand-designed acoustic tokenizer
(BEATs' k-means-on-fbank tokenizer).

Checkpoint/package verified 2026-08-15: `pip install -e .` from
github.com/habla-liaa/encodecmae (MIT-licensed, confirmed via the HF
model card for `lpepino/encodecmae-large-st`), no dependency conflicts
found (encodec==0.1.1 is Meta's standalone codec package, a different
namespace from transformers.EncodecModel already used elsewhere in this
project -- confirmed they coexist fine). One-time setup:
`pip install -e $AUDIO_COMP_EXTERNAL/encodecmae` after cloning the repo
there (see scripts/setup_encodecmae.sh).

Model name is "ec-ec-large_st", NOT "large-st" as the HF repo's own
usage snippet suggests -- verified directly by catching load_model()'s
own error message listing every valid name, rather than guessing a
close variant. Native sample rate 24kHz (confirmed via
model.wav_encoder.fs), matching its EnCodec-24khz backbone.

extract_features_from_array() returns per-frame activations
(1, n_frames, 1024) for the 'large' variant -- mean-pooled here the same
simple way as every other adapter's pooling convention.

**Short-clip edge case, caught by this project's mandatory standalone-
60ms-clip smoke test (2026-08-15), same category as the audio_jepa
kaldi.fbank and panns_cnn14 pooling incidents this checklist item exists
to catch**: extract_features_from_array()'s internal chunking loop only
processes windows with `> min_length` (2048) samples; a 60ms clip at
24kHz is 1,440 samples, so the loop produces zero chunks and
`acts[0]` raises IndexError on an empty list. Fixed by zero-padding any
waveform under MIN_LENGTH_SAMPLES before calling the model, rather than
letting it crash -- real probe-set clips are all much longer than this,
but the pipeline must not crash on the short end of the distribution.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import resample

MIN_LENGTH_SAMPLES = 2048  # matches encodecmae's own extract_features_from_array default


@register_model("encodecmae")
class EncodecMAEEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="encodecmae",
        hf_id="lpepino/encodecmae-large-st (github.com/habla-liaa/encodecmae, model name 'ec-ec-large_st')",
        paradigm="masked autoencoding, neural-codec-derived discrete target (general audio)",
        license="MIT",
        expected_sample_rate=24000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        from encodecmae import load_model

        self._model = load_model("ec-ec-large_st", device=self.device)

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        embeddings = []
        for w in resampled:
            w = w.astype(np.float32)
            if len(w) <= MIN_LENGTH_SAMPLES:
                w = np.pad(w, (0, MIN_LENGTH_SAMPLES + 1 - len(w)))
            feats = self._model.extract_features_from_array(w)
            embeddings.append(feats.mean(axis=1)[0])
        return np.stack(embeddings, axis=0)
