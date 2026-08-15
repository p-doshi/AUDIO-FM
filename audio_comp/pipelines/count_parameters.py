"""Verified parameter counts for every active model, by direct instantiation
-- the model object itself is the primary source, no ambiguity from
secondary-reported figures (which can differ by train-time vs.
inference-time counts, e.g. Audio-JEPA's own 96.7M/85.4M split). Only
loads each model (CPU, no forward pass/inference needed) -- cheap enough
to run as one sequential CPU job rather than per-model GPU jobs.

Usage:
    python -m audio_comp.pipelines.count_parameters --out results/model_parameter_counts.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]


def main(out_csv: str) -> None:
    from audio_comp.models import get_model_class

    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        active_models = sorted(yaml.safe_load(f)["active_models"])

    rows = []
    for name in active_models:
        print(f"[{name}] loading...", flush=True)
        try:
            adapter = get_model_class(name)(device="cpu")
            adapter.load()
            if name == "whisper":
                # Encoder-only, not the full encoder+decoder count -- the
                # decoder is never used for embeddings (see whisper.py's
                # never-touch-the-decoder policy), so counting it would
                # overstate whisper's effective size relative to every
                # encoder-only model in this roster. This is the
                # inference-time/embedding-extraction convention this
                # script uses throughout (direct instantiation naturally
                # gives inference-time counts for every other model
                # already, since only what's actually used gets loaded).
                n_params = sum(p.numel() for p in adapter._model.encoder.parameters())
                note = "encoder-only (decoder unused)"
            else:
                n_params = sum(p.numel() for p in adapter._model.parameters())
                note = ""
            rows.append({"model": name, "n_params": n_params, "note": note})
            print(f"[{name}] {n_params:,} parameters", flush=True)
        except Exception as e:  # noqa: BLE001 -- report and continue, don't let one model kill the whole sweep
            rows.append({"model": name, "n_params": "", "note": f"FAILED: {e}"})
            print(f"[{name}] FAILED: {e}", flush=True)
        # free memory before the next (potentially large) model
        del adapter

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["model", "n_params", "note"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "model_parameter_counts.csv"))
    args = parser.parse_args()
    main(args.out)
