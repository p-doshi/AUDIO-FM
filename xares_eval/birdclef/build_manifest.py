"""Download `mteb/birdclef25-mini` (a curated, balanced 1000-clip subset
of BirdCLEF 2025 -- 50 species x 20 recordings each, sourced from
Xeno-canto and iNaturalist) and build a manifest with a recording-level
fold assignment.

Unlike DeepShip, this dataset's row->audio->label mapping is not in
question -- each row's `url` field is unique across all 1000 rows
(confirmed 2026-08-10), so there's no join-ambiguity problem here. The
`vessel identity` -> `session` distinction that forced DeepShip into a
weaker file-level split doesn't apply the same way: every row already
*is* one independent field recording (one author, one place/time), so
grouping by row (recording) for the fold split is both the correct and
the strongest available leakage control, not a fallback.

Recordings vary hugely in length (1.0s to 491s, median 21s -- these are
real field recordings, not pre-trimmed clips like ESC-50/UrbanSound8K).
5s clips (not 10s, unlike DeepShip/FMA elsewhere in this project) --
bird vocalizations are typically much shorter events than ship engine
tonals, and BirdCLEF's own official evaluation protocol uses 5s
windows, so this matches domain convention rather than being an
arbitrary choice. Recordings shorter than 5s are zero-padded up to 5s
rather than dropped (unlike DeepShip's remainder-dropping) -- 69/1000
recordings are under 5s, and since each species only has 20 recordings
total, dropping them outright would meaningfully shrink some species'
already-small sample count. Recordings >=5s are chopped into
non-overlapping 5s windows with the tail remainder dropped, same as
DeepShip.

Each species has exactly 20 recordings, so a 5-fold split (4
recordings/species/fold) is clean and balanced -- no greedy-balancing
heuristic needed like DeepShip's uneven class counts required; fold
assignment is still deterministic (species-sorted, then round-robin by
recording index) for reproducibility.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import soundfile as sf

NUM_FOLDS = 5
CLIP_LENGTH_S = 5
HF_DATASET = "mteb/birdclef25-mini"


def build_rows(audio_out_dir: Path) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET, split="train")
    audio_out_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic per-species round-robin fold assignment: sort rows
    # within each species by their dataset index (arbitrary but fixed),
    # assign fold = position_within_species % NUM_FOLDS.
    species_seen: dict[str, int] = {}
    rows = []
    for i in range(len(ds)):
        row = ds[i]
        species = row["primary_label"]
        pos = species_seen.get(species, 0)
        species_seen[species] = pos + 1
        fold = pos % NUM_FOLDS

        wav_path = audio_out_dir / f"{species}_{i:04d}.wav"
        if not wav_path.exists():
            samples = row["recording"].get_all_samples()
            audio = samples.data.numpy()
            if audio.ndim > 1:
                audio = audio.mean(axis=0)
            else:
                audio = audio.squeeze()
            sf.write(wav_path, audio, samples.sample_rate)

        info = sf.info(wav_path)
        duration_s = info.frames / info.samplerate
        n_clips = max(1, int(duration_s // CLIP_LENGTH_S)) if duration_s >= CLIP_LENGTH_S else 1

        rows.append(
            dict(
                species=species,
                file=str(wav_path),
                row_index=i,
                common_name=row["common_name"],
                collection=row["collection"],
                author=row["author"],
                duration_s=round(duration_s, 3),
                n_clips=n_clips,
                fold=fold,
            )
        )
    return rows


def main(out_csv: str, audio_out_dir: str) -> None:
    rows = build_rows(Path(audio_out_dir))

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [
        "species",
        "file",
        "row_index",
        "common_name",
        "collection",
        "author",
        "duration_s",
        "n_clips",
        "fold",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    from collections import defaultdict

    totals = defaultdict(lambda: defaultdict(int))
    n_species_per_fold = defaultdict(set)
    for r in rows:
        totals[r["fold"]]["clips"] += r["n_clips"]
        totals[r["fold"]]["recordings"] += 1
        n_species_per_fold[r["fold"]].add(r["species"])
    print(f"Wrote {len(rows)} recordings ({sum(r['n_clips'] for r in rows)} clips at {CLIP_LENGTH_S}s) -> {out_csv}")
    for fold in range(NUM_FOLDS):
        print(
            f"  fold {fold}: {totals[fold]['recordings']} recordings, "
            f"{totals[fold]['clips']} clips, {len(n_species_per_fold[fold])} species"
        )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", default="data/birdclef_manifest.csv")
    parser.add_argument(
        "--audio-out-dir",
        default=os.path.join(
            os.environ.get("SCRATCH", os.path.expanduser("~/scratch")), "audio_comp", "birdclef_audio"
        ),
    )
    args = parser.parse_args()
    main(args.out_csv, args.audio_out_dir)
