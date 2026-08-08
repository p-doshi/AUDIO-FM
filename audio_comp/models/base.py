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


@dataclass(frozen=True)
class ModelInfo:
    name: str
    hf_id: str
    paradigm: str
    license: str
    expected_sample_rate: int


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
