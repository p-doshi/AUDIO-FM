#!/bin/bash
# One-time setup for the `audio_jepa` model adapter (audio_comp/models/audio_jepa.py).
# Clones LudovicTuncay/Audio-JEPA for its model-definition source
# (src/models/components/vision_transformer.py). The checkpoint itself
# (JEPA.ckpt, ~350MB) is fetched separately via huggingface_hub inside the
# adapter's load(), same as any other HF-hosted model.
set -euo pipefail

EXTERNAL_DIR="${AUDIO_COMP_EXTERNAL:-$HOME/audio_comp_external}"
REPO_DIR="$EXTERNAL_DIR/audio-jepa"

mkdir -p "$EXTERNAL_DIR"

if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 https://github.com/LudovicTuncay/Audio-JEPA.git "$REPO_DIR"
else
    echo "Audio-JEPA repo already present at $REPO_DIR, skipping clone"
fi

echo "Audio-JEPA source ready at $REPO_DIR (set AUDIO_COMP_EXTERNAL=$EXTERNAL_DIR if not using the default)"
