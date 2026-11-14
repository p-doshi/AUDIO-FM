"""Extract embeddings for a SUBSET of the probe set (e.g. a newly-added
category) and merge into an existing per-model .npz, rather than
re-running the full manifest through extract_embeddings.py -- avoids
wasteful re-extraction of clips a model already has embeddings for.

Usage:
    python -m audio_comp.pipelines.extract_embeddings_incremental \
        --model clap --category machine_sounds \
        --existing $SCRATCH/audio_comp/embeddings/clap.npz \
        --out $SCRATCH/audio_comp/embeddings/clap.npz
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
    parser.add_argument("--model", required=True)
    parser.add_argument("--category", required=True, help="only extract clips from this probe-set category")
    parser.add_argument("--manifest", type=Path, default=REPO_ROOT / "data" / "probe_set_manifest.csv")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path(os.environ.get("AUDIO_COMP_DATA_ROOT", os.path.expanduser("~/audio_comp_data"))),
    )
    parser.add_argument("--existing", type=Path, required=True, help="existing .npz to merge into")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    with open(args.manifest) as f:
        rows = [r for r in csv.DictReader(f) if r["category"] == args.category]
    if not rows:
        raise RuntimeError(f"no manifest rows found for category '{args.category}'")

    existing = np.load(args.existing, allow_pickle=True)
    existing_ids = set(existing["clip_ids"])
    new_rows = [r for r in rows if r["clip_id"] not in existing_ids]
    print(f"{len(rows)} clips in category '{args.category}', {len(new_rows)} not yet embedded")

    if not new_rows:
        print("nothing new to extract, existing file unchanged")
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = get_model_class(args.model)(device=device)
    encoder.load()

    new_ids, new_embeds = [], []
    for row in new_rows:
        waveform, sr = sf.read(args.data_root / row["path"], dtype="float32", always_2d=False)
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=1)
        new_embeds.append(encoder.embed(waveform, sr))
        new_ids.append(row["clip_id"])

    merged_ids = np.concatenate([existing["clip_ids"], np.array(new_ids)])
    merged_embeds = np.concatenate([existing["embeddings"], np.stack(new_embeds)], axis=0)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.out, clip_ids=merged_ids, embeddings=merged_embeds)
    print(f"wrote {len(merged_ids)} total embeddings ({merged_embeds.shape[1]}-d) to {args.out}")


if __name__ == "__main__":
    main()
