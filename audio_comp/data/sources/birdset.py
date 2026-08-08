"""BirdSet eval/soundscape subset — Bird sounds category. Pre-segmented
clips, CC-BY-4.0/CC0. Deliberately uses an eval/soundscape config, not the
CC-BY-NC Xeno-Canto-derived training subset — see
DBD-research-group/BirdSet on HF Datasets for the full split list.
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

import numpy as np

from ..base import BaseDatasetSource, Clip, DatasetInfo
from ..registry import register_dataset


@register_dataset("birdset_eval")
class BirdSetEvalSource(BaseDatasetSource):
    info = DatasetInfo(
        name="birdset_eval",
        category="bird_sounds",
        location="DBD-research-group/BirdSet (HF Datasets), 'HSN' eval config",
        license="CC-BY-4.0 / CC0 (eval/soundscape subset)",
        native_clip_seconds=5.0,
    )

    # HSN (High Sierra Nevada) is one of BirdSet's smaller pre-segmented
    # soundscape eval configs — swap for another eval config if unavailable.
    hf_config = "HSN"

    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        from datasets import load_dataset

        ds = load_dataset("DBD-research-group/BirdSet", self.hf_config, split="test")
        indices = list(range(len(ds)))
        random.Random(seed).shuffle(indices)
        for i in indices:
            example = ds[i]
            audio = example["audio"]
            waveform = np.asarray(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            yield Clip(
                clip_id=f"birdset_eval/{self.hf_config}/{i}",
                waveform=waveform,
                sample_rate=sr,
                source_dataset=self.info.name,
                category=self.info.category,
                duration_sec=len(waveform) / sr,
            )
