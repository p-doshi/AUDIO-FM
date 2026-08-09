"""UrbanSound8K (HF mirror) — City/urban noise category. Pre-segmented ≤4s
clips covering car horns, engine idling, jackhammer/drilling, sirens, etc.
— a direct match for "cars, construction". License on this community mirror
is listed as CC-BY-NC-4.0; verify against the original NYU/Kaggle release
before any redistribution use.
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

from ..base import BaseDatasetSource, Clip, DatasetInfo
from ..registry import register_dataset
from ._util import iter_hf_audio_clips


@register_dataset("urbansound8k")
class UrbanSound8KSource(BaseDatasetSource):
    info = DatasetInfo(
        name="urbansound8k",
        category="city_noise",
        location="danavery/urbansound8K (HF Datasets, unofficial mirror)",
        license="CC-BY-NC-4.0 (per mirror — verify against original)",
        native_clip_seconds=4.0,
    )

    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        from datasets import load_dataset

        ds = load_dataset("danavery/urbansound8K", split="train")
        indices = list(range(len(ds)))
        random.Random(seed).shuffle(indices)
        yield from iter_hf_audio_clips(
            ds, indices, lambda i: f"urbansound8k/{i}", self.info.category, self.info.name
        )
