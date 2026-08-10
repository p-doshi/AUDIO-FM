"""Common interface every audio foundation model adapter must implement.

To add a new model: subclass BaseAudioEncoder, set `info`, implement `load()`
and `embed_batch()`, decorate the class with `@register_model("name")` in your
adapter module, import that module from `audio_comp/models/__init__.py`, and
add a matching entry to `configs/models.yaml`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Sequence

import numpy as np

# Checkpoint-provenance labels (added 2026-08-10, per the model-roster
# expansion plan in CLAUDE.md's "Stage 2" section). Every registered model
# must declare one of these — see registry.py's register_model() for the
# schema check, and get_model_class() for the comparison-eligibility gate.
#
# - official_open_weights: checkpoint published by the model's own authors
#   (or their organization), under an explicit, unambiguous open license
#   covering the weights (not just the surrounding code).
# - official_public_weights_license_unclear: checkpoint published by the
#   model's own authors and publicly downloadable, but the license
#   covering the weights specifically is not explicitly stated (e.g. only
#   a repo-wide code license exists with no separate model-weights
#   statement) — usable, but flagged for a follow-up check before treating
#   it as unambiguously clear.
# - community_conversion: not published by the original authors — a
#   third-party port/conversion (e.g. an unofficial HF re-upload).
# - code_only: no usable checkpoint at all; only training/inference code
#   is available.
CHECKPOINT_STATUSES = (
    "official_open_weights",
    "official_public_weights_license_unclear",
    "community_conversion",
    "code_only",
)

# Only these statuses may be used in the main RSA/CKA comparison (enforced
# in registry.get_model_class()) — community_conversion and code_only
# models can still be registered (so the framework already models them,
# same convention as `beats` sitting in configs/models.yaml's
# deferred_models before its loader was wired up) but must be kept out of
# configs/models.yaml's active_models list.
COMPARISON_ELIGIBLE_CHECKPOINT_STATUSES = frozenset(
    {"official_open_weights", "official_public_weights_license_unclear"}
)


@dataclass(frozen=True)
class ModelInfo:
    name: str
    hf_id: str
    paradigm: str
    license: str
    expected_sample_rate: int
    checkpoint_status: str


class BaseAudioEncoder(ABC):
    """Adapter contract: load a checkpoint, embed batches of waveforms.

    Embeddings are pooled to one fixed-size vector per clip (mean-pool over
    time, by convention — see `audio_comp.models._util.mean_pool`) so the
    output of `embed_batch` can be fed directly into `audio_comp.geometry.rdm`.
    """

    info: ModelInfo

    def __init__(self, device: str = "cpu"):
        self.device = device
        self._model = None

    @abstractmethod
    def load(self) -> None:
        """Download (if needed) and load the checkpoint onto self.device."""

    @abstractmethod
    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        """Embed a batch of mono waveforms sampled at `sample_rate`.

        Returns an array of shape (len(waveforms), embedding_dim). Resampling
        to `self.info.expected_sample_rate` is the adapter's responsibility.
        """

    def embed(self, waveform: np.ndarray, sample_rate: int) -> np.ndarray:
        return self.embed_batch([waveform], sample_rate)[0]
