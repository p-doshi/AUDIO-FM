#!/bin/bash
#SBATCH --account=def-spadon_cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=compare_models
#SBATCH --output=/scratch/pdoshi/audio_comp/slurm_logs/%x-%j.out
#
# RSA/CKA/TwoNN comparison across all extracted embeddings. Submit with:
#   sbatch scripts/slurm/compare_models.sbatch
#
# Previously run as a plain background `nohup` process directly on the
# login node -- worked (if slowly) for the 9- and 11-model runs, but got
# silently killed within ~2.5 minutes at 12 models (2026-08-10), almost
# certainly the login node's own reaper/cgroup limits catching a CPU/
# memory-heavy background process, which is exactly the kind of shared-
# resource abuse those limits exist to prevent. Should have been a batch
# job from the start, same as every other compute-heavy step in this
# project -- fixed here rather than continuing to fight the login node.
set -euo pipefail

module load python/3.11 cuda/12.6 gcc arrow/25.0.0
source "${SCRATCH:-$HOME/scratch}/audio-comp-venv/bin/activate"

cd "$HOME/audio_comp"
export PYTHONPATH="$HOME/audio_comp:${PYTHONPATH:-}"

python -m audio_comp.pipelines.compare_models \
    --embeddings-dir "${SCRATCH:-$HOME/scratch}/audio_comp/embeddings" \
    --out-dir results
