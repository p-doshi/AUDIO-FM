"""amphion/SingVERSE — Noisy-music category. **Domain-narrower than FMA's
genre-diverse `music` category -- singing-voice only, not a full noisy
counterpart** (flagged honestly per this project's standing convention
rather than overstated; a genuinely genre-diverse, real-world-noisy,
clearly-licensed music dataset at FMA's scale was searched for and not
found, see journal.md 2026-08-18). Real (not synthetically mixed)
noisy/clean singing-voice pairs recorded across 19 real acoustic scenarios
(concert halls, roadsides, KTV rooms, restaurants, etc.) on professional
and non-professional devices. Only the `noisy_audio` side of each pair is
used here -- the `clean_audio` side exists in the source dataset but isn't
loaded, since the point of this source is background noise, not a
clean/noisy contrast pair (that pairing could be revisited later if a
matched clean-vs-noisy comparison is ever wanted).

CC-BY-4.0, verified directly against the dataset card
(huggingface.co/datasets/amphion/SingVERSE) via HfApi, not assumed.
3,929 paired clips, 18.14 total hours, native 44.1kHz.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from ..base import BaseDatasetSource, Clip, DatasetInfo
from ..registry import register_dataset


@register_dataset("singverse_noisy")
class SingVerseNoisySource(BaseDatasetSource):
    info = DatasetInfo(
        name="singverse_noisy",
        category="music_noisy",
        location="amphion/SingVERSE (HF Datasets, 'noisy_audio' field only)",
        license="CC-BY-4.0",
        native_clip_seconds=None,
    )

    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        from datasets import load_dataset  # deferred: heavy import, only needed here

        ds = load_dataset("amphion/SingVERSE", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=2000)
        for i, example in enumerate(ds):
            try:
                audio = example["noisy_audio"]
                waveform = np.asarray(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
            except Exception:
                continue
            if len(waveform) == 0:
                continue
            singer = example.get("singer", "na")
            song = example.get("song", "na")
            part = example.get("part", i)
            yield Clip(
                clip_id=f"singverse_noisy/{singer}_{song}_{part}_{i}",
                waveform=waveform,
                sample_rate=sr,
                source_dataset=self.info.name,
                category=self.info.category,
                duration_sec=len(waveform) / sr,
            )
