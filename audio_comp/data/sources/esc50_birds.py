"""ESC-50 (bird-related classes) — Bird sounds category. Pre-segmented 5s
clips, filtered to the "chirping_birds" and "crow" categories.

Replaced an earlier attempt to use DBD-research-group/BirdSet: that repo
only loads via a custom Python loading script (`BirdSet.py`), and
`datasets>=5` dropped script-based dataset loading entirely (not a
version-pinning issue — there's no config that avoids it). `ashraq/esc50` is
a script-less, Parquet-native mirror, hence the swap. Canonical ESC-50
license is CC-BY-NC-3.0 — verify against this mirror before redistribution
use. Only two of ESC-50's 50 classes are bird-related, so unlike the other
sources this one won't scale to thousands/category on its own — treat that
as a known limitation to revisit if Phase 1 proceeds past the pilot.
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

import numpy as np

from ..base import BaseDatasetSource, Clip, DatasetInfo
from ..registry import register_dataset

BIRD_CATEGORIES = {"chirping_birds", "crow"}


@register_dataset("esc50_birds")
class Esc50BirdsSource(BaseDatasetSource):
    info = DatasetInfo(
        name="esc50_birds",
        category="bird_sounds",
        location="ashraq/esc50 (HF Datasets), filtered to chirping_birds/crow",
        license="CC-BY-NC-3.0 (canonical ESC-50 license — verify against mirror)",
        native_clip_seconds=5.0,
    )

    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        from datasets import load_dataset

        ds = load_dataset("ashraq/esc50", split="train")
        indices = [i for i, category in enumerate(ds["category"]) if category in BIRD_CATEGORIES]
        random.Random(seed).shuffle(indices)
        for i in indices:
            example = ds[i]
            audio = example["audio"]
            waveform = np.asarray(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            yield Clip(
                clip_id=f"esc50_birds/{example['filename']}",
                waveform=waveform,
                sample_rate=sr,
                source_dataset=self.info.name,
                category=self.info.category,
                duration_sec=len(waveform) / sr,
            )
