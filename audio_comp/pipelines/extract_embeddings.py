"""Extract embeddings for ONE model across the full probe set.

Usage:
    python -m audio_comp.pipelines.extract_embeddings --model clap \
        --out $SCRATCH/audio_comp/embeddings/clap.npz

Run once per model — scripts/slurm/submit_all.sh submits one Slurm job per
model in parallel for this (the "distributed across GPUs" piece). Clips can
come from different source datasets with different native sample rates, so
each clip is embedded individually rather than batched — fine at pilot
scale (100 clips); revisit if the probe set scales to thousands and this
becomes the bottleneck.
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from audio_comp.models import get_model_class

REPO_ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="registered model name, e.g. 'clap'")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "probe_set_manifest.csv")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("AUDIO_COMP_DATA_ROOT", os.path.expanduser("~/audio_comp_data"))),
        help="base dir manifest 'path' entries are relative to",
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = get_model_class(args.model)(device=device)
    encoder.load()

    clip_ids, embeddings = [], []
    for row in rows:
        waveform, sr = sf.read(args.data_root / row["path"], dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        embeddings.append(encoder.embed(waveform, sr))
        clip_ids.append(row["clip_id"])

    embeddings = np.stack(embeddings)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, clip_ids=np.array(clip_ids), embeddings=embeddings)
    print(f"wrote {len(clip_ids)} embeddings ({embeddings.shape[1]}-d) to {args.out}")


if __name__ == "__main__":
    main()
