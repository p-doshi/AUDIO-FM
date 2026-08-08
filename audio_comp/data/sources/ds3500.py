"""DS3500 (ShipsEar-derived) — Ship/vessel category. Pre-segmented 5s clips,
CC-BY. See peng7554/DS3500 on HF Datasets. DeepShip (larger, more diverse)
is a stretch add-on for later — its GitHub hosting is only partial and the
rest requires emailing the author.
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

import numpy as np

from ..base import BaseDatasetSource, Clip, DatasetInfo
from ..registry import register_dataset


@register_dataset("ds3500")
class Ds3500Source(BaseDatasetSource):
    info = DatasetInfo(
        name="ds3500",
        category="ship_vessel",
        location="peng7554/DS3500 (HF Datasets)",
        license="CC-BY",
        native_clip_seconds=5.0,
    )

    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        from datasets import load_dataset

        # verification_mode="no_checks": this repo's embedded split metadata
        # (expected 4446 examples) doesn't match the actual data (4474) —
        # a mismatch on the dataset's own end, not fixable from our side.
        ds = load_dataset("peng7554/DS3500", split="train", verification_mode="no_checks")
        indices = list(range(len(ds)))
        random.Random(seed).shuffle(indices)
        for i in indices:
            example = ds[i]
            audio = example["audio"]
            waveform = np.asarray(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            yield Clip(
                clip_id=f"ds3500/{i}",
                waveform=waveform,
                sample_rate=sr,
                source_dataset=self.info.name,
                category=self.info.category,
                duration_sec=len(waveform) / sr,
            )
