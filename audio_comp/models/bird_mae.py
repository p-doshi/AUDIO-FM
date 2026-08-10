"""Bird-MAE — reconstruction-target (masked autoencoder), domain-specific
bioacoustic model. Fills two gaps in CLAUDE.md's Stage 2 plan at once:
the reconstruction-target paradigm axis (raw spectrogram reconstruction,
not a latent-space prediction target like data2vec/JEPA) and the
domain-specific-vs-general axis (trained on BirdSet, bird vocalizations
only) — also directly serves the bird_sounds / BirdCLEF track
(`xares_eval/birdclef/`), so this is one integration effort, not two.

Checkpoint provenance checked 2026-08-10 against the primary source
(github.com/DBD-research-group/Bird-MAE, arXiv 2504.12880 "Can Masked
Autoencoders Also Listen to Birds?"): DBD-research-group is the actual
paper's research group (same group behind the BirdSet dataset already
referenced elsewhere in this project), so this is a genuine official
release, not a third-party conversion — but **no LICENSE file exists in
the GitHub repo (confirmed via the GitHub API's license endpoint
returning 404, not just "undetected") and no license field is set on the
HF model card**. Unlike `beats`/`panns_cnn14` (where an actual repo-wide
LICENSE file exists and simply doesn't carve out the weights
separately), there's no license text here at all to lean on — hence
`checkpoint_status="official_public_weights_license_unclear"`, not
`official_open_weights`. Usable (this status is comparison-eligible per
base.py), but a real "verify before use" flag, not a formality — email
the authors before this checkpoint is used anywhere beyond this
internal research comparison.

Uses the model's own bundled `trust_remote_code=True` feature extractor
(`BirdMAEFeatureExtractor`) and model class (`BirdMAEModel`) rather than
reimplementing its fbank/padding/normalization logic — that logic is
non-trivial (fixed 512-frame target length, dataset-specific
mean/std normalization) and reimplementing it would risk subtly
diverging from what the checkpoint was actually trained on, the same
class of mistake the audio_jepa kaldi.fbank work was careful to avoid.
`BirdMAEModel.forward()` already mean-pools over the patch/time axis
internally (`config.global_pool == "mean"`) and returns the pooled
768-d vector as `last_hidden_state` — despite the name, it is NOT a
per-frame sequence here, no further pooling needed in embed_batch().

**Version-skew shim, needed and verified 2026-08-10**: the checkpoint's
own config.json records `transformers_version: "4.38.0"`, but this
project's venv has transformers 5.15.0 — a major-version gap, and
`modeling_bird_mae.py`'s `BirdMAEModel.__init__` never calls
`self.post_init()` (confirmed: the string "post_init" does not appear
anywhere in the file at all), which transformers 5.x now requires to
populate `self.all_tied_weights_keys` before
`_finalize_model_loading()` runs — without it, `AutoModel.from_pretrained`
crashes with `AttributeError: 'BirdMAEModel' object has no attribute
'all_tied_weights_keys'` before the model ever loads. This is an
upstream compatibility gap in the checkpoint's own vendored code, not
anything specific to this project's setup. Same category of fix as
audio_jepa's flash_attn shim: a narrow, documented monkeypatch on
`PreTrainedModel._move_missing_keys_from_meta_to_device` that defaults
the missing attribute to `{}` if absent, active only for the duration
of `load()`.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import torch
from transformers import AutoFeatureExtractor, AutoModel
from transformers.modeling_utils import PreTrainedModel

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import resample


def _install_missing_tied_weights_shim() -> None:
    if getattr(PreTrainedModel._move_missing_keys_from_meta_to_device, "_birdmae_shimmed", False):
        return
    _orig = PreTrainedModel._move_missing_keys_from_meta_to_device

    def _patched(self, *args, **kwargs):
        if not hasattr(self, "all_tied_weights_keys"):
            self.all_tied_weights_keys = {}
        return _orig(self, *args, **kwargs)

    _patched._birdmae_shimmed = True
    PreTrainedModel._move_missing_keys_from_meta_to_device = _patched


@register_model("bird_mae")
class BirdMAEEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="bird_mae",
        hf_id="DBD-research-group/Bird-MAE-Base",
        paradigm="masked autoencoding, reconstruction-target (bioacoustic, domain-specific)",
        license="unstated — no LICENSE file in the GitHub repo, no license field on the HF card (checked 2026-08-10)",
        expected_sample_rate=32000,
        checkpoint_status="official_public_weights_license_unclear",
    )

    def load(self) -> None:
        _install_missing_tied_weights_shim()
        self._extractor = AutoFeatureExtractor.from_pretrained(
            self.info.hf_id, trust_remote_code=True
        )
        self._model = (
            AutoModel.from_pretrained(self.info.hf_id, trust_remote_code=True)
            .to(self.device)
            .eval()
        )

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        max_len = max(len(w) for w in resampled)
        batch = np.stack([np.pad(w, (0, max_len - len(w))) for w in resampled]).astype(np.float32)
        features = self._extractor(batch, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self._model(input_values=features)
        return outputs.last_hidden_state.cpu().numpy()
