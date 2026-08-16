"""BEATs — masked-modeling general-audio model (Chen et al. 2022, arXiv
2212.09058), using self-distilled acoustic tokenizers as the MLM target.

**Engineering gap closed 2026-08-15** (was previously a stub -- see the
CLAUDE.md Stage 2 entry: license was already verified as MIT-covered,
this was purely "needs its own loader"). Vendors 3 source files from
github.com/microsoft/unilm/tree/master/beats (BEATs.py, backbone.py,
modules.py -- confirmed by reading BEATs.py's own imports that
Tokenizers.py/quantizer.py are only needed for training the acoustic
tokenizer, not for using a pretrained encoder). No fairseq/hydra
dependency, unlike audio_jepa's vendored code -- modules.py is fully
self-contained (torch only), confirmed by reading its imports directly.

Checkpoint: "BEATs_iter3+ (AS2M)", the *Pre-Trained Model* column entry
for that iteration row in the README's table -- verified directly against
the table's column structure (Tokenizer | Pre-Trained Model | AudioSet
Fine-Tuned Model 1 | AudioSet Fine-Tuned Model 2) that this is genuinely
self-supervised-only, NOT one of the two separate AudioSet-fine-tuned
checkpoints in that same row. The "(AS2M)" in the name describes which
iteration/tokenizer-training-data-scale this is, not a fine-tuning label
set -- a real ambiguity that was checked twice against the primary
source rather than assumed either way, per this project's standing
verify-before-use rule.

**Setup requires one manual step, no way around it**: BEATs' checkpoints
are OneDrive personal-share links, which return HTTP 403 to a plain
curl/wget (verified 2026-08-15 -- no programmatic download path exists
without an interactive browser session, unlike every other model in this
roster). Run `scripts/setup_beats.sh` first, which vendors the source
code automatically and prints the manual download instructions for the
one file it can't fetch itself.

Input representation: raw waveform in, but BEATs computes a 128-bin
mel-filterbank (kaldi.fbank) internally before its patch-embedding conv
-- same category as audio_jepa/audiomae's fbank-based pipelines, not a
Wav2Vec2FeatureExtractor-style raw-waveform-through model. Confirmed by
reading BEATs.py's own `preprocess()` method directly.

extract_features() returns (x, padding_mask) for a non-fine-tuned
checkpoint (predictor is None when cfg.finetuned_model=False) -- x is
the per-frame encoder output, mean-pooled here the same simple way
(no explicit mask-aware pooling) as every other adapter in this project's
_util.mean_pool() already does, for consistency rather than adding a
one-off exception.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import resample

EXTERNAL_DIR = Path(os.environ.get("AUDIO_COMP_EXTERNAL", os.path.expanduser("~/audio_comp_external")))
REPO_DIR = EXTERNAL_DIR / "beats"
CKPT_PATH = REPO_DIR / "BEATs_iter3_plus_AS2M.pt"


@register_model("beats")
class BeatsEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="beats",
        hf_id="github.com/microsoft/unilm (beats) -- BEATs_iter3+ (AS2M), Pre-Trained Model column",
        paradigm="masked modeling (general audio, self-distilled acoustic tokenizer target)",
        license="MIT (repo-wide; no separate per-checkpoint statement -- see module docstring)",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        if not (REPO_DIR / "BEATs.py").exists():
            raise RuntimeError(f"BEATs source not found at {REPO_DIR}. Run scripts/setup_beats.sh first.")
        if not CKPT_PATH.exists():
            raise RuntimeError(
                f"BEATs checkpoint not found at {CKPT_PATH}. Run scripts/setup_beats.sh for the manual "
                "download instructions (OneDrive can't be fetched by a script)."
            )
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))

        from BEATs import BEATs, BEATsConfig

        checkpoint = torch.load(str(CKPT_PATH), map_location="cpu", weights_only=False)
        cfg = BEATsConfig(checkpoint["cfg"])
        model = BEATs(cfg)
        model.load_state_dict(checkpoint["model"])
        self._model = model.to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        max_len = max(len(w) for w in resampled)
        padded = np.zeros((len(resampled), max_len), dtype=np.float32)
        padding_mask = np.zeros((len(resampled), max_len), dtype=bool)
        for i, w in enumerate(resampled):
            padded[i, : len(w)] = w
            if len(w) < max_len:
                padding_mask[i, len(w) :] = True

        source = torch.from_numpy(padded).to(self.device)
        mask = torch.from_numpy(padding_mask).to(self.device)
        with torch.no_grad():
            features, _ = self._model.extract_features(source, padding_mask=mask)
        return features.mean(dim=1).cpu().numpy()
