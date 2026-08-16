"""Parse the local MIMII 6dB-tier data into a manifest for a binary
anomaly-detection task (normal vs. abnormal), the dataset's own canonical
purpose. Fold = physical machine unit (16 total: 4 machine types x 4
unit IDs each), not individual clip -- MIMII's own paper used multiple
distinct physical units per machine type specifically to test
generalization across units, so grouping by unit ID is both the natural
leakage-safe split AND the one that matches how this dataset is meant to
be evaluated (same reasoning as ShipsEar's session-level grouping, one
level cleaner here since "physical unit" is an unambiguous, documented
grouping key rather than an inferred one).

Severe class imbalance is real, not smoothed over: 14,719 normal vs.
3,300 abnormal clips (matches the dataset's own documented totals
exactly, confirming the full 6dB tier downloaded correctly). Report
accuracy per class or via a balanced metric if this becomes the basis
for any claim beyond an aggregate accuracy number.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

MIMII_RAW_DIR = Path(os.environ.get("MIMII_RAW_DIR", "/scratch/pdoshi/audio_comp/mimii_raw"))
MACHINE_TYPES = ["valve", "pump", "fan", "slider"]
LABEL_MAP = {"normal": 0, "abnormal": 1}


def build_rows() -> list[dict]:
    rows = []
    for machine in MACHINE_TYPES:
        machine_dir = MIMII_RAW_DIR / machine
        if not machine_dir.exists():
            continue
        for id_dir in sorted(machine_dir.glob("id_*")):
            unit = f"{machine}_{id_dir.name}"
            for condition, label in LABEL_MAP.items():
                cond_dir = id_dir / condition
                if not cond_dir.exists():
                    continue
                for wav_path in sorted(cond_dir.glob("*.wav")):
                    rows.append(
                        dict(
                            file=str(wav_path),
                            machine=machine,
                            unit=unit,
                            condition=condition,
                            label=label,
                        )
                    )
    return rows


def main(out_csv: str) -> None:
    rows = build_rows()
    units = sorted({r["unit"] for r in rows})
    unit_to_fold = {u: i for i, u in enumerate(units)}
    for r in rows:
        r["fold"] = unit_to_fold[r["unit"]]

    os.makedirs(os.path.dirname(out_csv) or ".", exist_ok=True)
    fieldnames = ["file", "machine", "unit", "condition", "label", "fold"]
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    from collections import Counter

    condition_counts = Counter(r["condition"] for r in rows)
    print(f"Wrote {len(rows)} clips -> {out_csv}, {len(units)} folds (leave-one-machine-unit-out)")
    print(f"Class distribution: {dict(condition_counts)}")
    for unit in units:
        unit_rows = [r for r in rows if r["unit"] == unit]
        n_normal = sum(1 for r in unit_rows if r["condition"] == "normal")
        n_abnormal = sum(1 for r in unit_rows if r["condition"] == "abnormal")
        print(f"  fold {unit_to_fold[unit]} ({unit}): normal={n_normal} abnormal={n_abnormal}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out-csv", default="data/mimii_manifest.csv")
    args = parser.parse_args()
    main(args.out_csv)
