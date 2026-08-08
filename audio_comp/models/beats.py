"""BEATs — masked-modeling general-audio model.

Deferred from the kickoff run: no native `transformers.from_pretrained`
path. Official checkpoints (iter1/iter2/iter3/iter3+, AS20K/AS2M variants)
are direct-download links from
github.com/microsoft/unilm/tree/master/beats; loading needs BEATs' own
model code from that repo, and the checkpoint license isn't separately
restated from the code's — verify it before use. Wire this up as a
fast-follow once the pipeline is validated on the HF-native models.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model


@register_model("beats")
class BeatsEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="beats",
        hf_id="github.com/microsoft/unilm (beats) — no native HF repo",
        paradigm="masked modeling (general audio)",
        license="unverified for checkpoint weights — check github.com/microsoft/unilm before use",
        expected_sample_rate=16000,
    )

    def load(self) -> None:
        raise NotImplementedError(
            "BEATs has no transformers.from_pretrained path. Manually download a "
            "checkpoint from github.com/microsoft/unilm/tree/master/beats and vendor "
            "the BEATs model code before implementing this adapter."
        )

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        raise NotImplementedError("see load()")
