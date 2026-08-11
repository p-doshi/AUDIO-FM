#!/bin/bash
# One-time setup for the `audiomae` model adapter
# (audio_comp/models/audiomae.py). Downloads the official AudioMAE repo
# (facebookresearch/AudioMAE) via zip (not git clone -- see the
# 2026-08-10 journal entry on $HOME's intermittent Lustre I/O errors
# tripping on git's ref-log writes; zip has no ref-log) and the
# pretrained-only (not finetuned) ViT-B checkpoint from the repo's own
# README-linked Google Drive file, CC-BY-4.0 per the repo's LICENSE
# (verified 2026-08-10 directly against the repo, covers code and
# weights explicitly, no separate/contrary statement for the checkpoint).
set -euo pipefail

EXTERNAL_DIR="${AUDIO_COMP_EXTERNAL:-$HOME/audio_comp_external}"
REPO_DIR="$EXTERNAL_DIR/AudioMAE-main"
CKPT_PATH="$EXTERNAL_DIR/audiomae/pretrained.pth"

mkdir -p "$EXTERNAL_DIR" "$(dirname "$CKPT_PATH")"

if [ -d "$REPO_DIR" ]; then
    echo "AudioMAE repo already present at $REPO_DIR, skipping download"
else
    curl -fL -o "$EXTERNAL_DIR/audiomae_repo.zip" \
        https://github.com/facebookresearch/AudioMAE/archive/refs/heads/main.zip
    unzip -q "$EXTERNAL_DIR/audiomae_repo.zip" -d "$EXTERNAL_DIR"
    rm "$EXTERNAL_DIR/audiomae_repo.zip"
fi

if [ -f "$CKPT_PATH" ]; then
    echo "AudioMAE checkpoint already present at $CKPT_PATH, skipping download"
else
    python3 -m gdown "1ni_DV4dRf7GxM8k-Eirx71WP9Gg89wwu" -O "$CKPT_PATH"
fi

echo "AudioMAE ready: repo at $REPO_DIR, checkpoint at $CKPT_PATH"
