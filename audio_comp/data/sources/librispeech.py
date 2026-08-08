"""LibriSpeech ASR (English) — Speech category. Long-form audiobook
recordings, CC-BY-4.0. Streamed from HF Datasets. Segmented into random
`segment_seconds`-length windows per recording since native clips are much
longer than the other categories.

Note: this replaced an earlier attempt to pull an "english" config from
facebook/multilingual_librispeech — that dataset only has
dutch/french/german/italian/polish/portuguese/spanish configs; English
LibriSpeech is hosted separately (this dataset) rather than folded into
"multilingual" LibriSpeech. Also: Common Voice (the more obvious HF pick
historically) was pulled from HuggingFace in Oct 2025 and is now
Mozilla-Data-Collective-only, which is why this isn't using Common Voice.
"""
from __future__ import annotations

import random
from typing import Iterator, Optional

import numpy as np

from ..base import BaseDatasetSource, Clip, DatasetInfo
from ..registry import register_dataset

DEFAULT_SEGMENT_SECONDS = 5.0


@register_dataset("librispeech_en")
class LibriSpeechSource(BaseDatasetSource):
    info = DatasetInfo(
        name="librispeech_en",
        category="speech",
        location="openslr/librispeech_asr (HF Datasets, 'clean' config)",
        license="CC-BY-4.0",
        native_clip_seconds=None,
    )

    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        from datasets import load_dataset  # deferred: heavy import, only needed here

        segment_seconds = segment_seconds or DEFAULT_SEGMENT_SECONDS
        rng = random.Random(seed)
        ds = load_dataset("openslr/librispeech_asr", "clean", split="test", streaming=True)
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
                clip_id=f"librispeech_en/{example['id']}",
                waveform=segment,
                sample_rate=sr,
                source_dataset=self.info.name,
                category=self.info.category,
                duration_sec=segment_seconds,
            )
