"""Extract BirdNET embeddings for the full probe set.

**Architectural exception, not a bug**: unlike every other model in this
project, BirdNET has no official PyTorch/JAX/ONNX release -- only
TensorFlow (Keras/.pb/TFLite). Rather than adding TensorFlow to the main
`audio-comp-venv` (real CUDA/cuDNN version-conflict risk alongside the
existing torch+cuda12.6 setup -- the same class of risk the flash_attn/
torch incident already taught this project to avoid), BirdNET runs
**fully isolated** in its own venv (`$SCRATCH/birdnet-venv`, CPU-only,
zero shared dependencies with the main environment) and this script is
the entire integration surface: read the probe set manifest, extract
embeddings, write a `.npz` in the exact schema `extract_embeddings.py`
produces (`clip_ids`, `embeddings`) so every downstream pipeline
(`compare_models.py`, `alignment_uniformity_check.py`,
`breadth_hypothesis_check.py`) picks it up transparently by just
globbing the embeddings directory -- no changes needed on the PyTorch
side at all. This script is NOT part of the `audio_comp` package and is
never imported from the main venv; it can only be run with
`$SCRATCH/birdnet-venv` activated.

BirdNET has no `audio_comp.models.birdnet` adapter and is not
`@register_model`-registered -- it cannot be, since `BaseAudioEncoder`
subclasses assume a torch-loadable model in the same process. It is
therefore also not covered by `registry.py`'s `checkpoint_status` gate;
its provenance is documented here and in CLAUDE.md instead. Checkpoint
provenance verified 2026-08-10 directly against the primary source
(github.com/kahst/BirdNET-Analyzer, Kahl et al.): code MIT, model
weights CC BY-NC-SA 4.0 (non-commercial research use, which this project
is, is unrestricted; this is a real, more restrictive license than most
of the roster -- comparable to mert/music2vec's CC-BY-NC-4.0 already in
the table -- flag it the same way if this checkpoint's outputs are ever
used beyond this internal comparison).

Model download: `birdnet_analyzer.utils.ensure_model_exists()`'s default
URL (tuc.cloud) returned a 404 when tried 2026-08-10 -- downloaded the
official V2.4 TFLite release from Zenodo (zenodo.org/records/15050749,
`BirdNET_v2.4_tflite.zip`) directly instead and pointed `cfg.MODEL_PATH`
at the extracted `audio-model.tflite`, bypassing the package's broken
downloader rather than patching it.

Usage (inside $SCRATCH/birdnet-venv, NOT the main venv):
    python scripts/birdnet_extract_embeddings.py \
        --manifest data/probe_set_manifest.csv \
        --data-root ~/audio_comp_data \
        --out $SCRATCH/audio_comp/embeddings/birdnet.npz
"""
from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--model-path", required=True, help="path to audio-model.tflite")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    import birdnet_analyzer.config as cfg
    import birdnet_analyzer.model as model

    cfg.MODEL_PATH = args.model_path
    chunk_len = int(cfg.SIG_LENGTH * cfg.SAMPLE_RATE)

    import librosa

    with open(args.manifest) as f:
        rows = list(csv.DictReader(f))

    clip_ids, embeddings = [], []
    for i, row in enumerate(rows):
        path = Path(args.data_root).expanduser() / row["path"]
        wav, _ = librosa.load(path, sr=cfg.SAMPLE_RATE, mono=True)

        if len(wav) < chunk_len:
            wav = np.pad(wav, (0, chunk_len - len(wav)))
        n_chunks = max(1, len(wav) // chunk_len)
        chunks = [wav[c * chunk_len : (c + 1) * chunk_len].astype("float32") for c in range(n_chunks)]
        chunks = [c if len(c) == chunk_len else np.pad(c, (0, chunk_len - len(c))) for c in chunks]

        emb = model.embeddings(np.stack(chunks))
        pooled = emb.mean(axis=0)

        clip_ids.append(row["clip_id"])
        embeddings.append(pooled)

        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{len(rows)} clips done", flush=True)

    embeddings = np.stack(embeddings)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, clip_ids=np.array(clip_ids), embeddings=embeddings)
    print(f"wrote {len(clip_ids)} embeddings ({embeddings.shape[1]}-d) to {args.out}")


if __name__ == "__main__":
    main()
