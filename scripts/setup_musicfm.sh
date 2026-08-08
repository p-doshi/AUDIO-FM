#!/bin/bash
# One-time setup for the `musicfm` model adapter (audio_comp/models/musicfm.py).
# Clones minzwon/musicfm and downloads the MSD-trained checkpoint (per the
# upstream README, MSD outperforms the FMA-trained variant).
set -euo pipefail

EXTERNAL_DIR="${AUDIO_COMP_EXTERNAL:-$HOME/audio_comp_external}"
REPO_DIR="$EXTERNAL_DIR/musicfm"

mkdir -p "$EXTERNAL_DIR"

if [ ! -d "$REPO_DIR" ]; then
    git clone https://github.com/minzwon/musicfm.git "$REPO_DIR"
else
    echo "musicfm repo already present at $REPO_DIR, skipping clone"
fi

mkdir -p "$REPO_DIR/data"
wget -nc -P "$REPO_DIR/data/" https://huggingface.co/minzwon/MusicFM/resolve/main/msd_stats.json
wget -nc -P "$REPO_DIR/data/" https://huggingface.co/minzwon/MusicFM/resolve/main/pretrained_msd.pt

echo "MusicFM ready at $REPO_DIR (set AUDIO_COMP_EXTERNAL=$EXTERNAL_DIR if not using the default)"
