"""AMI Meeting Corpus (single distant mic) — Noisy-speech category, a
noisy counterpart to the existing clean `librispeech_en` speech source
(see CLAUDE.md's Known Risks: probe-set categories varied widely in
background-noise character, unflagged until a 2026-08-18 user question).
Real (not synthetically augmented) meeting-room acoustics -- cross-talk,
non-native speakers, real room echo/noise, recorded live across multiple
UK/Netherlands sites. The `sdm` (single distant microphone) config is used
deliberately over the closer-mic `ihm` config: `sdm` captures real room
noise/reverb far more than a headset mic does -- the whole point of this
source is noise, not just a second speech corpus.

CC-BY-4.0, verified directly against the dataset card
(huggingface.co/datasets/edinburghcstr/ami) via HfApi, not assumed.
267,303 pre-segmented utterance-level clips total (a few seconds each on
average) -- already utterance-length, no windowing/segmentation needed
unlike librispeech.py's long-form source.
"""
from __future__ import annotations

from typing import Iterator, Optional

import numpy as np

from ..base import BaseDatasetSource, Clip, DatasetInfo
from ..registry import register_dataset


@register_dataset("ami_meetings")
class AmiMeetingsSource(BaseDatasetSource):
    info = DatasetInfo(
        name="ami_meetings",
        category="speech_noisy",
        location="edinburghcstr/ami (HF Datasets, 'sdm' config)",
        license="CC-BY-4.0",
        native_clip_seconds=None,  # varies per utterance
    )

    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        from datasets import load_dataset  # deferred: heavy import, only needed here

        ds = load_dataset("edinburghcstr/ami", "sdm", split="train", streaming=True)
        ds = ds.shuffle(seed=seed, buffer_size=2000)
        for example in ds:
            try:
                audio = example["audio"]
                waveform = np.asarray(audio["array"], dtype=np.float32)
                sr = audio["sampling_rate"]
            except Exception:
                continue
            if len(waveform) == 0:
                continue
            yield Clip(
                clip_id=f"ami_meetings/{example['audio_id']}",
                waveform=waveform,
                sample_rate=sr,
                source_dataset=self.info.name,
                category=self.info.category,
                duration_sec=len(waveform) / sr,
            )
