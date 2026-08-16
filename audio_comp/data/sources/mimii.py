"""MIMII — industrial machine condition sounds (valve/pump/fan/slide-rail,
normal vs. anomalous), the project's 6th probe-set category and the first
domain none of the 19 roster models were ever trained on (verified via
model_report.csv's category breakdown, 2026-08-14 -- 9 speech, 3 music, 2
bird_sounds, 5 general/AudioSet-broad, zero on anything mechanical/
industrial). Chosen over two other verified candidates (InsectSet459,
ICBHI 2017 respiratory sounds) for its clean, uniform format and
unambiguous license -- see journal.md, 2026-08-15, for the full
comparison.

Source: Zenodo record 3384388 (CC-BY-SA 4.0, confirmed via the Zenodo API
directly, not just the record's landing page). Only the cleanest SNR tier
(6dB) was downloaded -- the probe set only needs a few thousand clips,
not the full 18,019-clip/100GB three-tier sweep. Raw files must already
be extracted locally (see scripts/download_mimii.sh) since Zenodo has no
HF Datasets-style streaming API; this source reads from local disk, not
a remote dataset id, hence `location` below is a path convention, not an
HF/Zenodo identifier the way every other source module's is.

Native format already uniform (16kHz, 10s clips, mono) -- no segmentation
needed, same simplicity as ds3500.py.
"""
from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Iterator, Optional

import soundfile as sf

from ..base import BaseDatasetSource, Clip, DatasetInfo
from ..registry import register_dataset

MIMII_RAW_DIR = Path(os.environ.get("MIMII_RAW_DIR", "/scratch/pdoshi/audio_comp/mimii_raw"))
MACHINE_TYPES = ["valve", "pump", "fan", "slider"]


@register_dataset("mimii")
class MimiiSource(BaseDatasetSource):
    info = DatasetInfo(
        name="mimii",
        category="machine_sounds",
        location=f"local: {MIMII_RAW_DIR} (from Zenodo 3384388, 6dB tier only)",
        license="CC-BY-SA 4.0",
        native_clip_seconds=10.0,
    )

    def iter_clips(self, seed: int, segment_seconds: Optional[float] = None) -> Iterator[Clip]:
        if not MIMII_RAW_DIR.exists():
            raise RuntimeError(
                f"MIMII raw data not found at {MIMII_RAW_DIR}. Download the 6dB-tier zips from "
                "Zenodo record 3384388 and extract first (see journal.md, 2026-08-15)."
            )

        paths = []
        for machine in MACHINE_TYPES:
            machine_dir = MIMII_RAW_DIR / machine
            if not machine_dir.exists():
                continue
            for id_dir in sorted(machine_dir.glob("id_*")):
                for condition in ("normal", "abnormal"):
                    cond_dir = id_dir / condition
                    if not cond_dir.exists():
                        continue
                    for wav_path in sorted(cond_dir.glob("*.wav")):
                        paths.append((machine, id_dir.name, condition, wav_path))

        random.Random(seed).shuffle(paths)

        for machine, machine_id, condition, wav_path in paths:
            waveform, sr = sf.read(str(wav_path), dtype="float32")
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            clip_id = f"mimii/{machine}_{machine_id}_{condition}_{wav_path.stem}"
            yield Clip(
                clip_id=clip_id,
                waveform=waveform,
                sample_rate=sr,
                source_dataset=self.info.name,
                category=self.info.category,
                duration_sec=len(waveform) / sr,
            )
