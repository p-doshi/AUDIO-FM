#!/bin/bash
# One-time setup for the `panns_cnn14` model adapter
# (audio_comp/models/panns_cnn14.py). Downloads the official Cnn14
# checkpoint from its authors' own Zenodo record (verified 2026-08-10
# against github.com/qiuqiangkong/audioset_tagging_cnn directly, not a
# secondary source: repo is MIT-licensed (LICENSE.MIT), checkpoints
# hosted at https://zenodo.org/record/3987831 by the same authors).
#
# Deliberately NOT using the `panns_inference` pip package's own
# download-to-$HOME-on-first-use behavior -- same reasoning as every
# other model in this project: raw checkpoints belong on $SCRATCH/
# $AUDIO_COMP_EXTERNAL, not $HOME (see the 2026-08-10 $HOME storage
# incident in journal.md).
set -euo pipefail

EXTERNAL_DIR="${AUDIO_COMP_EXTERNAL:-$HOME/audio_comp_external}"
CKPT_DIR="$EXTERNAL_DIR/panns"
CKPT_PATH="$CKPT_DIR/Cnn14_mAP=0.431.pth"

mkdir -p "$CKPT_DIR"

if [ -f "$CKPT_PATH" ]; then
    echo "PANNs Cnn14 checkpoint already present at $CKPT_PATH, skipping download"
else
    curl -fL -o "$CKPT_PATH" \
        'https://zenodo.org/record/3987831/files/Cnn14_mAP%3D0.431.pth?download=1'
fi

echo "PANNs Cnn14 ready at $CKPT_PATH (set AUDIO_COMP_EXTERNAL=$EXTERNAL_DIR if not using the default)"
