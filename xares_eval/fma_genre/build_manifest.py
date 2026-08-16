"""Build a local-file manifest for FMA-genre classification, for the
LoRA-vs-frozen matched fine-tuning comparison. Unlike UrbanSound8K/BirdCLEF,
this needs no new download -- reuses the same local FMA-small audio cache
this project's probe set already uses (`audio_comp/data/sources/fma.py`'s
RAW_DIR), joined against `fma_metadata/tracks.csv`'s `genre_top` field.

Matches X-ARES's own upstream `fma_genre_config`'s 8-class taxonomy
exactly (Hip-Hop/Pop/Folk/Experimental/Rock/International/Electronic/
Instrumental -- FMA-small's own standard, balanced 8-genre curation, not
a custom selection), so results are comparable to any existing FMA-genre
X-ARES frozen-probe numbers.

Fold assignment: FMA-small's own `tracks.csv` `set/split` column already
provides train/validation/test -- reused directly rather than inventing a
new split, matching "each dataset's own natural held-out split."
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

DATA_ROOT = Path(os.environ.get("AUDIO_COMP_DATA_ROOT", os.path.expanduser("~/audio_comp_data")))
RAW_DIR = DATA_ROOT / "raw" / "fma_small"
TRACKS_CSV = DATA_ROOT / "raw" / "fma_metadata" / "tracks.csv"
OUT_MANIFEST = Path(os.environ.get("FMA_GENRE_MANIFEST_CSV", "data/fma_genre_manifest.csv"))

GENRE_CLASSES = {
    "Hip-Hop", "Pop", "Folk", "Experimental", "Rock", "International", "Electronic", "Instrumental",
}


def load_genre_and_split(tracks_csv: Path) -> dict[str, tuple[str, str]]:
    """track_id (zero-padded to 6 digits, matching fma_small filenames) ->
    (genre_top, split)."""
    with open(tracks_csv, newline="") as f:
        reader = csv.reader(f)
        top_header = next(reader)
        sub_header = next(reader)
        next(reader)  # third header row (all blank / "track_id")

        genre_idx = next(i for i, (t, s) in enumerate(zip(top_header, sub_header)) if s == "genre_top")
        split_idx = next(i for i, (t, s) in enumerate(zip(top_header, sub_header)) if s == "split")

        out = {}
        for row in reader:
            if not row or not row[0]:
                continue
            track_id = row[0].zfill(6)
            genre = row[genre_idx]
            split = row[split_idx]
            if genre in GENRE_CLASSES:
                out[track_id] = (genre, split)
        return out


def main() -> None:
    genre_split = load_genre_and_split(TRACKS_CSV)
    mp3_paths = sorted(RAW_DIR.rglob("*.mp3"))
    print(f"{len(mp3_paths)} local mp3s, {len(genre_split)} tracks with a valid 8-class genre_top")

    rows = []
    for path in mp3_paths:
        track_id = path.stem
        if track_id not in genre_split:
            continue
        genre, split = genre_split[track_id]
        rows.append(dict(file=str(path), track_id=track_id, genre=genre, split=split))

    OUT_MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_MANIFEST, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "track_id", "genre", "split"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} clips -> {OUT_MANIFEST}")

    import collections

    print("genre counts:", collections.Counter(r["genre"] for r in rows))
    print("split counts:", collections.Counter(r["split"] for r in rows))


if __name__ == "__main__":
    main()
