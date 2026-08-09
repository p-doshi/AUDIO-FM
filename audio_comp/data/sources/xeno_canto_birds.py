"""tglcourse/5s_birdcall_samples_top20 — Bird sounds category. Pre-segmented
5s clips (peak-picked from longer Xeno-Canto recordings), 20 species,
~9,595 clips total — enough headroom for the probe set to scale into the
thousands/category, unlike the earlier `esc50_birds` source it replaces
(ESC-50 only had 80 bird-related clips total across its two closest
classes, nowhere near enough once the probe set moved past the pilot).

License: listed as "Unknown" on the dataset card. Source is Xeno-Canto,
whose per-recording licenses default to the CC-BY-NC-SA family but vary by
contributor; the aggregator didn't carry per-clip license metadata through.
Treat as unverified/mixed CC-family and say so explicitly in any write-up —
do not treat this as confirmed CC-BY-NC-SA for every clip.
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

from ..base import BaseDatasetSource, Clip, DatasetInfo
from ..registry import register_dataset
from ._util import iter_hf_audio_clips


@register_dataset("xeno_canto_birds")
class XenoCantoBirdsSource(BaseDatasetSource):
    info = DatasetInfo(
        name="xeno_canto_birds",
        category="bird_sounds",
        location="tglcourse/5s_birdcall_samples_top20 (HF Datasets)",
        license="unverified/mixed (Xeno-Canto-derived, listed as 'Unknown' on the dataset card)",
        native_clip_seconds=5.0,
    )

    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        from datasets import load_dataset

        ds = load_dataset("tglcourse/5s_birdcall_samples_top20", split="train")
        indices = list(range(len(ds)))
        random.Random(seed).shuffle(indices)
        yield from iter_hf_audio_clips(
            ds, indices, lambda i: f"xeno_canto_birds/{i}", self.info.category, self.info.name
        )
