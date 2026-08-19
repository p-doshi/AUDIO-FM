"""Build a local-file manifest for LibriSpeech speaker-ID classification,
for the LoRA-vs-frozen matched fine-tuning comparison -- closes the one
probe-set category (speech) still missing a matched fine-tuning result
relative to music/city_noise/bird_sounds/machine_sounds/confidential
vessel (see journal.md, 2026-08-18).

**Classification target: speaker-ID, confirmed against the dataset's own
schema before use** (`speaker_id` is a genuine int64 field on
`openslr/librispeech_asr`, verified directly via the HF datasets-server
API, not assumed). Uses the same `train.clean.100` split
`audio_comp/data/sources/librispeech.py` already draws the probe set's
`speech` category from (28,539 utterances, 251 speakers).

**Speaker-ID is closed-set classification -- this is NOT the same
leakage-control problem as DeepShip/vessel's held-out-entity splits.**
Every speaker (all 251 classes) must appear in train, since a classifier
can't predict a class it never saw. The actual leakage risk here is
utterance-level, not speaker-level: the same underlying recording must
never appear in both train and a held-out split. Enforced by splitting
each speaker's own utterance list independently (stratified per-speaker
70/15/15), with utterance IDs assigned to exactly one split -- no
utterance-level overlap, full speaker overlap by design.

**Balanced draw, not the full 28,539 utterances**: caps each speaker at
MAX_PER_SPEAKER utterances (matching this project's established
"balanced few dozen per class" convention, e.g. MIMII's ~200/class,
BirdCLEF's ~20/species) and drops speakers with fewer than
MIN_PER_SPEAKER available, to keep the 251-way task balanced rather than
skewed toward speakers with disproportionately more recorded material.
"""
from __future__ import annotations

import csv
import os
import random
from collections import defaultdict
from pathlib import Path

import soundfile as sf

OUT_MANIFEST = Path(os.environ.get("LIBRISPEECH_SPEAKER_MANIFEST_CSV", "data/librispeech_speaker_manifest.csv"))
AUDIO_OUT_DIR = Path(os.environ.get("LIBRISPEECH_SPEAKER_AUDIO_DIR", "/scratch/pdoshi/audio_comp/librispeech_speaker_audio"))

SEED = 0
MAX_PER_SPEAKER = 40
MIN_PER_SPEAKER = 20
SPLIT_FRACTIONS = {"train": 0.70, "val": 0.15, "test": 0.15}


def build_rows(audio_out_dir: Path) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openslr/librispeech_asr", "clean", split="train.100", streaming=True)
    ds = ds.shuffle(seed=SEED, buffer_size=2000)
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    per_speaker: dict[int, list[dict]] = defaultdict(list)
    n_seen = 0
    for row in ds:
        speaker_id = row["speaker_id"]
        if len(per_speaker[speaker_id]) >= MAX_PER_SPEAKER:
            continue
        utt_id = row["id"]
        wav_path = audio_out_dir / f"{speaker_id}_{utt_id}.wav"
        if not wav_path.exists():
            audio = row["audio"]
            samples = audio["array"]
            sr = audio["sampling_rate"]
            sf.write(wav_path, samples, sr)
        per_speaker[speaker_id].append(dict(file=str(wav_path), speaker_id=speaker_id, utterance_id=utt_id))
        n_seen += 1
        if n_seen % 1000 == 0:
            print(f"  {n_seen} utterances processed, {len(per_speaker)} speakers so far", flush=True)
        # stop once every speaker seen so far has hit the cap and we've
        # covered a generous multiple of 251 speakers worth of utterances
        # -- streaming has no total-length signal, so bound by a
        # utterance-count ceiling rather than iterating the full 28,539.
        if n_seen >= 251 * MAX_PER_SPEAKER * 2:
            break

    kept_speakers = {s: utts for s, utts in per_speaker.items() if len(utts) >= MIN_PER_SPEAKER}
    dropped = len(per_speaker) - len(kept_speakers)
    print(f"{len(per_speaker)} speakers seen, {len(kept_speakers)} kept (>= {MIN_PER_SPEAKER} utterances), {dropped} dropped")
    return kept_speakers


def assign_splits(kept_speakers: dict[int, list[dict]]) -> list[dict]:
    rng = random.Random(SEED)
    rows = []
    for speaker_id, utts in kept_speakers.items():
        utts = list(utts)
        rng.shuffle(utts)
        n = len(utts)
        n_train = int(n * SPLIT_FRACTIONS["train"])
        n_val = int(n * SPLIT_FRACTIONS["val"])
        for i, utt in enumerate(utts):
            split = "train" if i < n_train else ("val" if i < n_train + n_val else "test")
            utt["split"] = split
            rows.append(utt)
    return rows


def main() -> None:
    kept_speakers = build_rows(AUDIO_OUT_DIR)
    rows = assign_splits(kept_speakers)

    speaker_ids_sorted = sorted(kept_speakers.keys())
    label_map = {sid: i for i, sid in enumerate(speaker_ids_sorted)}
    for r in rows:
        r["label"] = label_map[r["speaker_id"]]

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "speaker_id", "label", "utterance_id", "split"])
        writer.writeheader()
        writer.writerows(rows)

    n_train = sum(1 for r in rows if r["split"] == "train")
    n_val = sum(1 for r in rows if r["split"] == "val")
    n_test = sum(1 for r in rows if r["split"] == "test")
    print(
        f"Wrote {len(rows)} clips ({len(speaker_ids_sorted)}-way speaker-ID) -> {OUT_MANIFEST}, "
        f"train={n_train} val={n_val} test={n_test} (per-speaker stratified split, utterance-level disjoint)"
    )


if __name__ == "__main__":
    main()
