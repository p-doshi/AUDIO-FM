"""Extract the "Results:" table from a run_xares.sbatch log and append it
to results/xares_results.csv. The raw logs (tens of MB of tqdm progress
bar spam) aren't checked into git — this is the durable record instead.

Usage:
    python -m xares_eval.parse_results <model_name> <log_path>
"""
from __future__ import annotations

import csv
import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_CSV = REPO_ROOT / "results" / "xares_results.csv"

ROW_RE = re.compile(r"^(.+?)\s{2,}([\d.]+)\s+([\d.]+)\s+(True|False)\s*$")


def parse_results(log_path: Path) -> list[dict]:
    text = log_path.read_text(errors="replace")
    marker = text.rfind("Results:")
    if marker == -1:
        raise ValueError(f"No 'Results:' table found in {log_path}")
    block = text[marker:].splitlines()

    rows = []
    for line in block[2:]:  # skip "Results:" and the header line
        m = ROW_RE.match(line)
        if not m:
            break
        task, mlp_score, knn_score, private = m.groups()
        rows.append(
            {
                "task": task.strip(),
                "mlp_score": float(mlp_score),
                "knn_score": float(knn_score),
                "private": private,
            }
        )
    return rows


def main() -> None:
    model_name, log_path = sys.argv[1], Path(sys.argv[2])
    rows = parse_results(log_path)
    if not rows:
        raise ValueError(f"No result rows parsed from {log_path}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    file_exists = OUT_CSV.exists()
    with open(OUT_CSV, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "task", "mlp_score", "knn_score", "private"])
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow({"model": model_name, **row})

    print(f"appended {len(rows)} rows for {model_name} to {OUT_CSV}")


if __name__ == "__main__":
    main()
