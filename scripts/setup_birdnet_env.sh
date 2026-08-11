#!/bin/bash
# One-time setup for the BirdNET embedding-extraction environment.
# BirdNET has no official PyTorch/JAX/ONNX release (TensorFlow only) --
# see scripts/birdnet_extract_embeddings.py's module docstring for why
# this lives in a fully separate, isolated venv rather than adding
# TensorFlow to the main audio-comp-venv.
set -euo pipefail

module load python/3.11
VENV_DIR="${SCRATCH:-$HOME/scratch}/birdnet-venv"
python3 -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip -q
pip install birdnet-analyzer -q   # pulls in tensorflow (CPU is sufficient --
                                    # this is a one-time batch extraction, not
                                    # a training workload; sidesteps any
                                    # CUDA/cuDNN version-matching risk entirely)

CKPT_DIR="$VENV_DIR/lib/python3.11/site-packages/birdnet_analyzer/checkpoints"
mkdir -p "$CKPT_DIR"
if [ -f "$CKPT_DIR/audio-model.tflite" ]; then
    echo "BirdNET checkpoint already present, skipping download"
else
    # birdnet_analyzer's own ensure_model_exists() default URL
    # (tuc.cloud) returned a 404 as of 2026-08-10 -- using the official
    # Zenodo record (zenodo.org/records/15050749) directly instead.
    curl -fL -o /tmp/BirdNET_v2.4_tflite.zip \
        "https://zenodo.org/api/records/15050749/files/BirdNET_v2.4_tflite.zip/content"
    unzip -q /tmp/BirdNET_v2.4_tflite.zip -d "$CKPT_DIR"
    rm /tmp/BirdNET_v2.4_tflite.zip
fi

echo "BirdNET env ready at $VENV_DIR, checkpoint at $CKPT_DIR/audio-model.tflite"
echo "Run scripts/birdnet_extract_embeddings.py with this venv activated (NOT audio-comp-venv)."
