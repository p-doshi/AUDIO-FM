"""Multilingual LibriSpeech (English) — Speech category. Long-form audiobook
recordings, CC-BY-4.0. Streamed from HF Datasets (no separate download
script needed — full non-streaming pull would be ~44.5k hours). Segmented
into random `segment_seconds`-length windows per recording since native
clips are much longer than the other categories.

Note: Common Voice (the more obvious HF pick historically) was pulled from
HuggingFace in Oct 2025 and is now Mozilla-Data-Collective-only — MLS is
used instead specifically because it's still HF-native.
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

import numpy as np

from ..base import BaseDatasetSource, Clip, DatasetInfo
from ..registry import register_dataset

DEFAULT_SEGMENT_SECONDS = 5.0


@register_dataset("mls_english")
class MlsEnglishSource(BaseDatasetSource):
    info = DatasetInfo(
        name="mls_english",
        category="speech",
        location="facebook/multilingual_librispeech (HF Datasets, 'english' config)",
        license="CC-BY-4.0",
        native_clip_seconds=None,
    )

    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        from datasets import load_dataset  # deferred: heavy import, only needed here

        segment_seconds = segment_seconds or DEFAULT_SEGMENT_SECONDS
        rng = random.Random(seed)
        ds = load_dataset(
            "facebook/multilingual_librispeech", "english", split="test", streaming=True
        )
        ds = ds.shuffle(seed=seed, buffer_size=1000)
        for example in ds:
            audio = example["audio"]
            waveform = np.asarray(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
            seg_len = int(segment_seconds * sr)
            if len(waveform) <= seg_len:
                continue
            start = rng.randint(0, len(waveform) - seg_len)
            segment = waveform[start : start + seg_len]
            yield Clip(
                clip_id=f"mls_english/{example['id']}",
                waveform=segment,
                sample_rate=sr,
                source_dataset=self.info.name,
                category=self.info.category,
                duration_sec=segment_seconds,
            )
