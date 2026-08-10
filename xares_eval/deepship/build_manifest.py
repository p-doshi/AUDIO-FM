"""Parse DeepShip's per-class metafiles and the 63 GitHub-hosted wav files
into a single manifest with a file-grouped (not vessel-grouped) fold
assignment.

DeepShip's full dataset (265 ships, 47h04m) is only available by emailing
the author; `github.com/irfankamboh/DeepShip` hosts a 63-clip subset (12
cargo, 20 passengership, 28 tanker, 3 tug — see scripts/setup_deepship.sh).

**The metafile's `record_id` column is NOT a reliable join key to the
hosted wav filenames — confirmed 2026-08-10, this is a real limitation of
this partial dataset, not a bug in this script.** Metafiles have far more
rows than hosted files (e.g. tanker-metafile has 244 rows for 28 hosted
files) and `record_id` repeats within a metafile across entirely
different vessels/durations (e.g. tanker record `47` appears twice —
`CHERRY GALAXY` dur=246s and `SEAVOYAGER` dur=251s — while the actual
`Tanker/47.wav` measures 21s, matching neither). Systematic duration-based
matching across all 63 hosted files found: Cargo 10/12 mismatched,
Passengership 2/20, Tanker 15/28 (13 more only resolved by picking the
closest-duration duplicate, not a confirmed match), Tug 0/3 (n=3, not
informative). Whoever assembled the GitHub subset most likely renamed
files sequentially per class folder, independent of the original
metafile's record_id numbering.

**Practical consequence, decided with the user (2026-08-10): vessel
identity cannot be trusted for grouping.** `vessel_name`/`date`/`time`/
`range_field` below are kept only as best-effort, low-confidence
informational fields (see `metadata_confidence`) — NEVER used for the
fold split. The only structurally-guaranteed-correct facts about each
hosted file are its class (from the folder name) and its own audio
content. Fold grouping is therefore done at the **recording-file** level
(each of the 63 hosted wav files is its own group — matching the
"session" reading of the leakage-control requirement, not the stricter
"vessel instance across multiple sessions" reading, which this data
cannot support). This still blocks the main leakage failure mode (clips
cut from the same continuous recording pass split across train/test) but
cannot guarantee the same physical vessel never appears in two different
files across folds. State this caveat wherever DeepShip results are
reported.

`duration_s`/`n_clips` are computed from each file's own measured audio
length (via `soundfile`), never from the metafile's duration field — the
metafile field is unreliable for exactly the same reason record_id is.

3 folds (not 5, unlike ESC-50/GTZAN's k=5/k=10 convention elsewhere in
this project) because `tug` only has 3 hosted files — more folds would
guarantee some folds have zero tug examples. Even at k=3, tug has exactly
one file per fold: a real statistical-power limitation of this partial
dataset.

Fold assignment is deterministic (sorted by clip count, ties broken by
record_id; greedy-balanced across folds within each class) rather than
randomized, so no seed is needed for reproducibility.
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import soundfile as sf

NUM_FOLDS = 3
CLIP_LENGTH_S = 10
DURATION_MATCH_TOLERANCE_S = 1.0

CLASS_METAFILES = {
    "cargo": ("Cargo", "cargo-metafile"),
    "passengership": ("Passengership", "passengership-metafile"),
    "tanker": ("Tanker", "tanker-metafile"),
    "tug": ("Tug", "tug-metafile"),
}


def _parse_metafile(path: Path) -> dict[str, list[dict]]:
    """Returns record_id -> list of candidate rows (record_id repeats
    within a metafile, see module docstring — never assume one row)."""
    candidates: dict[str, list[dict]] = defaultdict(list)
    with open(path) as f:
        for line in f:
            parts = [p.strip() for p in line.strip().split(",")]
            if not parts or not parts[0]:
                continue
            rec_id, _node_id, vessel_name, date, time_, duration_s = parts[:6]
            range_field = parts[6] if len(parts) > 6 and parts[6] else None
            candidates[rec_id].append(
                dict(
                    vessel_name=vessel_name.strip(),
                    date=date,
                    time=time_,
                    duration_s=float(duration_s),
                    range_field=range_field,
                )
            )
    return candidates


def _best_metadata_match(actual_duration_s: float, candidates: list[dict]) -> tuple[dict, str]:
    """Pick the metafile candidate whose declared duration is closest to
    the file's real measured duration, and label how much to trust it."""
    if not candidates:
        return dict(vessel_name="", date="", time="", range_field=""), "no_candidates"

    best = min(candidates, key=lambda c: abs(c["duration_s"] - actual_duration_s))
    matched = abs(best["duration_s"] - actual_duration_s) <= DURATION_MATCH_TOLERANCE_S
    if len(candidates) == 1:
        confidence = "exact" if matched else "mismatch"
    else:
        confidence = "ambiguous_resolved" if matched else "mismatch"
    return best, confidence


def build_rows(deepship_root: Path) -> list[dict]:
    rows = []
    for vessel_class, (subdir, metafile_name) in CLASS_METAFILES.items():
        class_dir = deepship_root / subdir
        meta = _parse_metafile(class_dir / metafile_name)
        for wav_path in sorted(glob.glob(str(class_dir / "*.wav"))):
            rec_id = Path(wav_path).stem
            info = sf.info(wav_path)
            actual_duration_s = info.frames / info.samplerate
            n_clips = int(actual_duration_s // CLIP_LENGTH_S)

            best, confidence = _best_metadata_match(actual_duration_s, meta.get(rec_id, []))
            rows.append(
                dict(
                    vessel_class=vessel_class,
                    file=wav_path,
                    record_id=rec_id,
                    duration_s=round(actual_duration_s, 2),
                    n_clips=n_clips,
                    vessel_name=best["vessel_name"],
                    date=best["date"],
                    time=best["time"],
                    range_field=best["range_field"] or "",
                    metadata_confidence=confidence,
                )
            )
    return rows


def assign_folds(rows: list[dict]) -> None:
    """Greedy-balance individual recording files across NUM_FOLDS folds,
    within each class, by clip count. Each file is its own group (see
    module docstring for why vessel-level grouping isn't possible here).
    Mutates each row in place, adding 'fold'."""
    fold_class_clip_totals: dict[tuple[str, int], int] = defaultdict(int)
    for vessel_class in CLASS_METAFILES:
        class_rows = [r for r in rows if r["vessel_class"] == vessel_class]
        class_rows.sort(key=lambda r: (-r["n_clips"], r["record_id"]))
        for r in class_rows:
            target_fold = min(
                range(NUM_FOLDS),
                key=lambda f: (fold_class_clip_totals[(vessel_class, f)], f),
            )
            fold_class_clip_totals[(vessel_class, target_fold)] += r["n_clips"]
            r["fold"] = target_fold


def main(deepship_root: str, out_csv: str) -> None:
    rows = build_rows(Path(deepship_root))
    assign_folds(rows)

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = [
        "vessel_class",
        "file",
        "record_id",
        "duration_s",
        "n_clips",
        "fold",
        "vessel_name",
        "date",
        "time",
        "range_field",
        "metadata_confidence",
    ]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    totals = defaultdict(lambda: defaultdict(int))
    confidence_counts = defaultdict(int)
    for r in rows:
        totals[r["fold"]][r["vessel_class"]] += r["n_clips"]
        confidence_counts[r["metadata_confidence"]] += 1
    print(f"Wrote {len(rows)} recordings -> {out_csv}")
    print(f"Clip counts (at {CLIP_LENGTH_S}s/clip) by fold x class (file-level grouping):")
    for fold in range(NUM_FOLDS):
        print(f"  fold {fold}: {dict(totals[fold])}")
    print(f"vessel_name/date/range_field match confidence (informational only, NOT used for splitting): {dict(confidence_counts)}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--deepship-root",
        default=os.path.join(
            os.environ.get("AUDIO_COMP_EXTERNAL", os.path.expanduser("~/audio_comp_external")),
            "DeepShip",
            "DeepShip-main",
        ),
    )
    parser.add_argument("--out-csv", default="data/deepship_manifest.csv")
    args = parser.parse_args()
    main(args.deepship_root, args.out_csv)
