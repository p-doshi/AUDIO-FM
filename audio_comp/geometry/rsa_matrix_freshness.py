"""Data-freshness guard for `results/rsa_matrix.csv` (and its CKA/
intrinsic-dimension siblings), added 2026-08-20 after a real incident:
the committed matrix silently drifted one, then two, probe-set
expansions behind `data/probe_set_manifest.csv` (machine_sounds, then
AMI/SingVERSE), and every downstream consumer (`breadth_hypothesis_check.py`,
`scaling_analysis.py`, `fdr_correction.py`) kept computing "current"
numbers against a stale file with no way to notice. Reconstructing which
of three different cited numbers for the same claim was actually current
took a real audit -- this exists so that never has to happen silently
again.

`compare_models.py` writes a small sidecar `rsa_matrix.meta.json`
alongside the matrix, recording the probe-set manifest's own mtime/row
count at extraction time. Any script that reads `rsa_matrix.csv` should
call `check_freshness()` first and print (not raise -- a stale matrix is
usually still fine to look at, just not to cite as current) a loud
warning if the manifest has moved on since the matrix was generated.
"""
from __future__ import annotations

import json
from pathlib import Path


def write_freshness_meta(matrix_path: Path, manifest_path: Path) -> None:
    manifest_path = Path(manifest_path)
    with open(manifest_path) as f:
        n_rows = sum(1 for _ in f) - 1  # minus header
    meta = {
        "manifest_path": str(manifest_path),
        "manifest_mtime": manifest_path.stat().st_mtime,
        "manifest_row_count": n_rows,
    }
    meta_path = Path(matrix_path).with_name(Path(matrix_path).stem + ".meta.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)


def check_freshness(matrix_path: Path, manifest_path: Path) -> bool:
    """Returns True if fresh (or unknown -- no meta file, an older matrix
    predating this check), False and prints a loud warning if the
    manifest has grown/changed since the matrix was generated. Never
    raises -- callers decide whether to proceed."""
    matrix_path = Path(matrix_path)
    manifest_path = Path(manifest_path)
    meta_path = matrix_path.with_name(matrix_path.stem + ".meta.json")
    if not meta_path.exists():
        print(
            f"[freshness-check] WARNING: no {meta_path.name} found next to {matrix_path.name} -- "
            f"cannot verify this matrix reflects the current probe set. It may predate this "
            f"freshness-tracking system (added 2026-08-20). Re-run compare_models.py to generate one."
        )
        return True  # unknown, not a confirmed staleness -- don't false-alarm on old-but-fine matrices
    with open(meta_path) as f:
        meta = json.load(f)
    with open(manifest_path) as f:
        current_rows = sum(1 for _ in f) - 1
    current_mtime = manifest_path.stat().st_mtime

    stale = current_rows != meta["manifest_row_count"] or current_mtime > meta["manifest_mtime"]
    if stale:
        print(
            f"[freshness-check] *** WARNING: {matrix_path.name} is STALE ***\n"
            f"  It was generated against a probe set with {meta['manifest_row_count']} rows "
            f"(manifest mtime {meta['manifest_mtime']}).\n"
            f"  The current probe set has {current_rows} rows (manifest mtime {current_mtime}).\n"
            f"  Numbers computed from this matrix do NOT reflect the current probe set -- "
            f"re-run compare_models.py before citing anything as current."
        )
    return not stale
