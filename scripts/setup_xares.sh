#!/bin/bash
# One-time setup for Stage 1's X-ARES downstream evaluation
# (xares_eval/). The `xares` PyPI package (pip install "xares[examples]")
# only ships the framework code, not the task definitions
# (src/tasks/*.py) — those live only in the git repo.
set -euo pipefail

EXTERNAL_DIR="${AUDIO_COMP_EXTERNAL:-$HOME/audio_comp_external}"
REPO_DIR="$EXTERNAL_DIR/xares"

mkdir -p "$EXTERNAL_DIR"

if [ ! -d "$REPO_DIR" ]; then
    git clone --depth 1 https://github.com/jimbozhang/xares.git "$REPO_DIR"
else
    echo "xares repo already present at $REPO_DIR, skipping clone"
fi

echo "xares task definitions ready at $REPO_DIR/src/tasks/ (set AUDIO_COMP_EXTERNAL=$EXTERNAL_DIR if not using the default)"
