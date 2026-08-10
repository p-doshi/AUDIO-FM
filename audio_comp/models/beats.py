"""BEATs — masked-modeling general-audio model.

Deferred from the kickoff run: no native `transformers.from_pretrained`
path. Official checkpoints (iter1/iter2/iter3/iter3+, AS20K/AS2M variants)
are direct-download links from
github.com/microsoft/unilm/tree/master/beats; loading needs BEATs' own
model code from that repo.

**License, checked 2026-08-10** (per the checkpoint-provenance labeling
pass in CLAUDE.md's Stage 2 section): the unilm repo root LICENSE is a
standard MIT license ("Copyright (c) Microsoft Corporation"). The BEATs
subdirectory's own README license section says "This project is licensed
under the license found in the LICENSE file in the root directory of
this source tree" — no separate carve-out or distinct terms for the
checkpoint weights specifically, and no contrary/restrictive statement
anywhere either. Read as MIT covering the whole repo including the
checkpoints by absence of any separate statement, not because of an
explicit per-checkpoint license line — hence `checkpoint_status` below is
`official_open_weights` rather than `official_public_weights_license_unclear`,
but this is a judgment call, not an airtight citation; re-verify if this
ever becomes a redistribution question rather than just an internal-use one.

**As of 2026-08-10: this is purely an engineering gap, not a checkpoint-
availability blocker** — the checkpoint itself is fine to use, `beats`
just still needs its own loader (BEATs' model code isn't
`transformers.from_pretrained`-compatible, same category of work as
`musicfm`/`audio_jepa`'s custom loaders, which are both already wired up).
Worth revisiting given it's no longer blocked on anything external — not
implemented in this pass, which was scoped to registry/CLAUDE.md updates
only, not new adapter work.
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
        license="MIT (repo-wide; no separate per-checkpoint statement — see module docstring)",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        raise NotImplementedError(
            "BEATs has no transformers.from_pretrained path. Manually download a "
            "checkpoint from github.com/microsoft/unilm/tree/master/beats and vendor "
            "the BEATs model code before implementing this adapter."
        )

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        raise NotImplementedError("see load()")
