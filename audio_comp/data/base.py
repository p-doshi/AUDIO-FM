"""Common interface every probe-set dataset source must implement.

To add a new dataset/category: subclass BaseDatasetSource, set `info`,
implement `iter_clips()`, decorate the class with `@register_dataset("name")`
in your source module, import that module from
`audio_comp/data/sources/__init__.py`, and add a matching entry to
`configs/categories.yaml`.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Iterator, Optional

import numpy as np


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    category: str
    location: str  # HF dataset id, or a URL/instructions for non-HF sources
    license: str
    native_clip_seconds: Optional[float]  # None if long-form and needs segmentation


@dataclass(frozen=True)
class Clip:
    clip_id: str
    waveform: np.ndarray
    sample_rate: int
    source_dataset: str
    category: str
    duration_sec: float


class BaseDatasetSource(ABC):
    info: DatasetInfo

    @abstractmethod
    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        """Yield Clip objects in a deterministic (seeded) order.

        Sources whose native recordings are already short (pre-segmented)
        ignore `segment_seconds`. Long-form sources sample a random
        `segment_seconds`-length window per recording, seeded for
        reproducibility.
        """
