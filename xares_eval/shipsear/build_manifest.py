"""Parse the real ShipsEar archive (peng7554/DS3500's `ShipsEar.zip` on HF
Datasets -- the *real* recordings, not the synthetic ray-theory-simulated
DS3500.zip half) into a manifest for a 3-class vessel-type task.

**Why 3 classes, not the dataset's own 5 (A-E)** -- checked directly
against the zip's internal folder structure (`shipsear_5s_16k/<class>/
<session>/<file>.wav`) 2026-08-13, not assumed from the README's summary
table (whose own totals don't even match the actual zip contents: README
claims 1948 total, the zip actually has 2223 wav files -- logged here
since it's exactly the kind of doc-vs-primary-source mismatch this
project's standing verification rule exists to catch):
    class 0 (A): 369 files across 5 sessions (52/101/144/32/40)
    class 1 (B): 301 files across 3 sessions (196/26/79)
    class 2 (C): 843 files, ALL from ONE session -- zero within-class
                 diversity, a leakage-safe train/test split is impossible
    class 3 (D): 486 files across 2 sessions (186/300)
    class 4 (E): 224 files, ALL from ONE session (also not a vessel type
                 -- E is "environmental noise", i.e. no vessel present)
Classes 2 and 4 are dropped entirely: with only one recording session
each, any split that isn't leakage-safe would let a model "recognize" a
specific vessel's background acoustic fingerprint from adjacent time
slices of the *same* recording rather than genuinely generalizing, which
would produce an inflated, meaningless accuracy number -- worse than not
having the check at all, since it would look like validation while
actually testing nothing. This is explicitly a bare-minimum/non-research-
grade check (per the user, 2026-08-13) -- the real generalization test is
the confidential Stage 5 v2 data, not this.

Fold = session, not file (leave-one-session-out): the 10 remaining
sessions across classes 0/1/3 are all necessarily used as folds directly
-- this is the ShipsEar analogue of DeepShip's file-level grouping
decision (xares_eval/deepship/build_manifest.py), same reasoning, one
level up (session instead of file, since here the file *is* already the
smallest atomic unit and multiple files share a session).
"""
from __future__ import annotations

import csv
import os
import zipfile
from pathlib import Path

KEPT_CLASSES = {"0": "A", "1": "B", "3": "D"}  # class 2 (C) and 4 (E) dropped, see module docstring
CLASS_LABEL_NAMES = {"A": 0, "B": 1, "D": 2}  # dense 0..2 labels for the task itself


def build_rows(zip_path: Path) -> list[dict]:
    rows = []
    with zipfile.ZipFile(zip_path) as z:
        for name in z.namelist():
            if not name.endswith(".wav"):
                continue
            parts = name.split("/")
            raw_class, session = parts[1], parts[2]
            if raw_class not in KEPT_CLASSES:
                continue
            class_name = KEPT_CLASSES[raw_class]
            rows.append(
                dict(
                    zip_member=name,
                    class_name=class_name,
                    label=CLASS_LABEL_NAMES[class_name],
                    session=session,
                )
            )
    return rows


def main(zip_path: str, out_csv: str) -> None:
    rows = build_rows(Path(zip_path))
    sessions = sorted({r["session"] for r in rows})
    session_to_fold = {s: i for i, s in enumerate(sessions)}
    for r in rows:
        r["fold"] = session_to_fold[r["session"]]

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = ["zip_member", "class_name", "label", "session", "fold"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    from collections import defaultdict

    totals = defaultdict(lambda: defaultdict(int))
    for r in rows:
        totals[r["fold"]][r["class_name"]] += 1
    print(f"Wrote {len(rows)} clips -> {out_csv}, {len(sessions)} folds (leave-one-session-out)")
    for fold in sorted(totals):
        print(f"  fold {fold} (session {sessions[fold]}): {dict(totals[fold])}")


if __name__ == "__main__":
    import argparse

    from huggingface_hub import hf_hub_download

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", default="data/shipsear_manifest.csv")
    args = parser.parse_args()
    zip_path = hf_hub_download("peng7554/DS3500", "ShipsEar.zip", repo_type="dataset")
    main(zip_path, args.out_csv)
