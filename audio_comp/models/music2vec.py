"""music2vec — data2vec-family (self-distillation, EMA-updated teacher) music model.

Originally mislabeled "JEPA-family" in CLAUDE.md's candidate table and
carried through several turns of analysis before being corrected
(2026-08-09). data2vec's student encoder operates directly on masked input
and predicts the EMA teacher's averaged top-K layer representations — no
separate predictor network. JEPA (as A-JEPA/Audio-JEPA's own methods
sections describe it) has three components: context encoder, EMA target
encoder, and a *separate predictor network* conditioned on the context
representation. Both lineages descend from BYOL and are
self-distillation-with-EMA-teacher, but the decoupled predictor is the
actual dividing line, and music2vec is on the data2vec side of it. See the
correction note in CLAUDE.md for the full implication (H1, as originally
phrased, needs a second working JEPA-family checkpoint before it's
testable — music2vec doesn't supply one).

Upstream m-a-p/music2vec-v1's published config.json has `vocab_size` set to
a filesystem path (an artifact of the original author's local machine,
leaked into the published config) instead of an int — irrelevant to this
architecture's actual feature-extraction use, but `transformers`' now-strict
config field validation rejects it outright where older versions silently
ignored the type mismatch. Worked around by copying the snapshot to a
mutable local dir and patching just that field before loading.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from huggingface_hub import snapshot_download
from transformers import AutoModel, Wav2Vec2FeatureExtractor

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample

EXTERNAL_DIR = Path(os.environ.get("AUDIO_COMP_EXTERNAL", os.path.expanduser("~/audio_comp_external")))
PATCHED_DIR = EXTERNAL_DIR / "music2vec-v1-patched"


@register_model("music2vec")
class Music2VecEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="music2vec",
        hf_id="m-a-p/music2vec-v1",
        paradigm="data2vec-family (self-distillation, EMA-updated teacher; music)",
        license="CC-BY-NC-4.0",
        expected_sample_rate=16000,
    )

    def load(self) -> None:
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.info.hf_id)

        if not PATCHED_DIR.exists():
            snapshot_dir = snapshot_download(self.info.hf_id)
            shutil.copytree(snapshot_dir, PATCHED_DIR)
            config_path = PATCHED_DIR / "config.json"
            with open(config_path) as f:
                config = json.load(f)
            if not isinstance(config.get("vocab_size"), int):
                config["vocab_size"] = 32  # unused by this architecture; any int satisfies validation
                with open(config_path, "w") as f:
                    json.dump(config, f)

        self._model = AutoModel.from_pretrained(str(PATCHED_DIR)).to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        inputs = self._extractor(
            resampled, sampling_rate=self.info.expected_sample_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return mean_pool(outputs.last_hidden_state).cpu().numpy()
