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

# xares_eval/tasks is a symlink to the cloned repo's task definitions.
# It has to live inside xares_eval/ (not just be referenced by absolute
# path) because xares.run's attr_from_py_path() does a real import_module()
# on the path with "/" replaced by "." — an absolute path becomes an
# invalid leading-dot relative-import name. Not committed to git (machine-
# specific target), hence this symlink step instead of a checked-in path.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ln -sfn "$REPO_DIR/src/tasks" "$REPO_ROOT/xares_eval/tasks"

echo "xares task definitions ready at $REPO_ROOT/xares_eval/tasks/ (set AUDIO_COMP_EXTERNAL=$EXTERNAL_DIR if not using the default)"
