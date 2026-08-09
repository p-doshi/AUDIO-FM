"""Shared helper for sources that index a non-streaming `datasets.Dataset`
by position and decode an `audio` column per row.

Real-world audio datasets at thousands-of-clips scale reliably contain a
handful of corrupted/undecodable entries — invisible at a 20-clip pilot,
routinely hit once the probe set scales up (seen in FMA-small, DS3500).
Centralized here so every source skips bad entries the same way instead of
each one growing its own try/except.
"""
from __future__ import annotations

from typing import Callable, Iterator

import numpy as np

from ..base import Clip


def iter_hf_audio_clips(
    ds, indices, clip_id_fn: Callable[[int], str], category: str, source_name: str
) -> Iterator[Clip]:
    for i in indices:
        try:
            example = ds[i]
            audio = example["audio"]
            waveform = np.asarray(audio["array"], dtype=np.float32)
            sr = audio["sampling_rate"]
        except Exception:
            continue
        yield Clip(
            clip_id=clip_id_fn(i),
            waveform=waveform,
            sample_rate=sr,
            source_dataset=source_name,
            category=category,
            duration_sec=len(waveform) / sr,
        )
