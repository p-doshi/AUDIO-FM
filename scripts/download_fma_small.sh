#!/bin/bash
# One-time download for the `fma_small` dataset source (audio_comp/data/sources/fma.py).
# Official FMA release (github.com/mdeff/fma) — CC-BY-family licensed per-track.
set -euo pipefail

DATA_ROOT="${AUDIO_COMP_DATA_ROOT:-$HOME/audio_comp_data}"
RAW_DIR="$DATA_ROOT/raw"

mkdir -p "$RAW_DIR"
cd "$RAW_DIR"

if [ ! -f fma_small.zip ]; then
    wget https://os.unil.cloud.switch.ch/fma/fma_small.zip
fi
if [ ! -f fma_metadata.zip ]; then
    wget https://os.unil.cloud.switch.ch/fma/fma_metadata.zip
fi

if [ ! -d fma_small ]; then
    unzip -q fma_small.zip
fi
if [ ! -d fma_metadata ]; then
    unzip -q fma_metadata.zip
fi

echo "FMA-small ready at $RAW_DIR/fma_small (set AUDIO_COMP_DATA_ROOT=$DATA_ROOT if not using the default)"
