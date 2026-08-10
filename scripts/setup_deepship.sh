#!/bin/bash
# One-time setup for the DeepShip vessel-acoustics task
# (xares_eval/deepship/*.py, xares_eval/tasks_scratch/deepship_task.py).
#
# DeepShip's full ~265-vessel dataset requires emailing the author
# (mirfan@mail.nwpu.edu.cn); only a 63-clip, 4-class (cargo/passengership/
# tanker/tug) subset is hosted on GitHub. That subset is what this script
# fetches. Downloaded via GitHub's zip archive rather than `git clone` --
# `git clone` hit the same transient $HOME Lustre I/O error documented in
# the 2026-08-10 journal entry (git's ref-log append failed; plain file
# writes to the same directory succeeded), and a zip download sidesteps
# git's ref-log write entirely and has no .git history to go stale.
set -euo pipefail

EXTERNAL_DIR="${AUDIO_COMP_EXTERNAL:-$HOME/audio_comp_external}"
REPO_DIR="$EXTERNAL_DIR/DeepShip"

mkdir -p "$REPO_DIR"

if [ -d "$REPO_DIR/DeepShip-main" ]; then
    echo "DeepShip already present at $REPO_DIR/DeepShip-main, skipping download"
else
    curl -fL -o "$REPO_DIR/deepship.zip" \
        https://github.com/irfankamboh/DeepShip/archive/refs/heads/master.zip
    unzip -q "$REPO_DIR/deepship.zip" -d "$REPO_DIR"
    rm "$REPO_DIR/deepship.zip"
fi

echo "DeepShip ready at $REPO_DIR/DeepShip-main (set AUDIO_COMP_EXTERNAL=$EXTERNAL_DIR if not using the default)"
