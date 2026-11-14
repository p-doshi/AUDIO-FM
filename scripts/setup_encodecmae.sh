#!/bin/bash
# One-time setup for the `encodecmae` model adapter (audio_comp/models/encodecmae.py).
# Clones habla-liaa/encodecmae and pip-installs it editable. The actual
# checkpoint weights are fetched automatically inside load_model() the
# first time it runs (via huggingface_hub + a companion EnCodec checkpoint
# from Meta's own CDN) -- no manual download needed here, unlike beats.
set -euo pipefail

EXTERNAL_DIR="${AUDIO_COMP_EXTERNAL:-$HOME/audio_comp_external}"
REPO_DIR="$EXTERNAL_DIR/encodecmae"

mkdir -p "$EXTERNAL_DIR"

if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 https://github.com/habla-liaa/encodecmae.git "$REPO_DIR"
else
    echo "encodecmae repo already present at $REPO_DIR, skipping clone"
fi

pip install -e "$REPO_DIR"

echo "encodecmae ready. Checkpoint (~large-st) downloads automatically on first use."
