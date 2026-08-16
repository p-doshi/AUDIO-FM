#!/bin/bash
# One-time setup for the `beats` model adapter (audio_comp/models/beats.py).
#
# Vendors just the 3 source files actually needed for inference (BEATs.py,
# backbone.py, modules.py -- verified 2026-08-15 that Tokenizers.py/
# quantizer.py are only needed for training the acoustic tokenizer itself,
# not for using a pretrained BEATs encoder). No fairseq/hydra dependency --
# modules.py is fully self-contained (torch only), unlike audio_jepa's
# vendored code.
#
# The checkpoint CANNOT be fetched by this script -- BEATs' checkpoints are
# hosted on OneDrive personal-share links, which return 403 to a plain
# curl/wget (verified 2026-08-15, no programmatic download path exists
# without an interactive browser session). You must download it yourself:
#
#   1. Open this link in a browser: https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf
#      (this is "BEATs_iter3+ (AS2M)", the Pre-Trained Model column entry
#      for that iteration -- NOT one of the "AudioSet Fine-Tuned Model 1/2"
#      columns, which are separate, different checkpoints. Verified against
#      the README's table structure directly, not assumed from the
#      "(AS2M)" name alone -- that suffix names the iteration, not a
#      fine-tuning label set.)
#   2. Download the .pt file
#   3. Place it at: $AUDIO_COMP_EXTERNAL/beats/BEATs_iter3_plus_AS2M.pt
#      (default AUDIO_COMP_EXTERNAL is ~/audio_comp_external)
set -euo pipefail

EXTERNAL_DIR="${AUDIO_COMP_EXTERNAL:-$HOME/audio_comp_external}"
REPO_DIR="$EXTERNAL_DIR/beats"

mkdir -p "$REPO_DIR"

for f in BEATs.py backbone.py modules.py; do
    if [ ! -f "$REPO_DIR/$f" ]; then
        curl -sf "https://raw.githubusercontent.com/microsoft/unilm/master/beats/$f" -o "$REPO_DIR/$f"
        echo "fetched $f"
    else
        echo "$f already present, skipping"
    fi
done

echo
echo "BEATs source ready at $REPO_DIR"
echo
CKPT_PATH="$REPO_DIR/BEATs_iter3_plus_AS2M.pt"
if [ -f "$CKPT_PATH" ]; then
    echo "Checkpoint already present at $CKPT_PATH -- setup complete."
else
    echo "*** MANUAL STEP STILL NEEDED ***"
    echo "Download the checkpoint yourself (OneDrive can't be fetched by a script):"
    echo "  1. https://1drv.ms/u/s!AqeByhGUtINrgcpke6_lRSZEKD5j2Q?e=A3FpOf"
    echo "  2. Save the downloaded .pt file to: $CKPT_PATH"
fi
