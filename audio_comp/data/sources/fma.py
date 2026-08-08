"""FMA-small — Music category, sourced from the official FMA release
(github.com/mdeff/fma), CC-BY-family licensed per-track. Chosen over GTZAN
for license cleanliness. Run `scripts/download_fma_small.sh` once before
using this source (~8.5GB: fma_small.zip + fma_metadata.zip).

fma_small has 8,000 30s clips — the same cache this pilot pulls 20 from is
reused unchanged when the probe set scales to thousands/category later.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterator, Optional

import librosa

from ..base import BaseDatasetSource, Clip, DatasetInfo
from ..registry import register_dataset

DATA_ROOT = Path(os.environ.get("AUDIO_COMP_DATA_ROOT", os.path.expanduser("~/audio_comp_data")))
RAW_DIR = DATA_ROOT / "raw" / "fma_small"


@register_dataset("fma_small")
class FmaSmallSource(BaseDatasetSource):
    info = DatasetInfo(
        name="fma_small",
        category="music",
        location="https://github.com/mdeff/fma (fma_small.zip, official release)",
        license="CC-BY family (per-track, see fma_metadata/tracks.csv)",
        native_clip_seconds=30.0,
    )

    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        if not RAW_DIR.exists():
            raise RuntimeError(
                f"FMA-small not found at {RAW_DIR}. Run scripts/download_fma_small.sh first."
            )
        mp3_paths = sorted(RAW_DIR.rglob("*.mp3"))
        if not mp3_paths:
            raise RuntimeError(f"No .mp3 files found under {RAW_DIR} — download may be incomplete.")
        rng = random.Random(seed)
        rng.shuffle(mp3_paths)
        for path in mp3_paths:
            waveform, sr = librosa.load(path, sr=None, mono=True)
            yield Clip(
                clip_id=f"fma_small/{path.stem}",
                waveform=waveform,
                sample_rate=sr,
                source_dataset=self.info.name,
                category=self.info.category,
                duration_sec=len(waveform) / sr,
            )
