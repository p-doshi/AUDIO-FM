#!/bin/bash
#SBATCH --account=def-spadon_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --job-name=xares_eval
#SBATCH --output=/scratch/pdoshi/audio_comp/slurm_logs/%x-%j.out
#
# X-ARES Stage 1 downstream evaluation for one model against the 3
# overlapping tasks (FMA-genre, LibriSpeech-ASR, UrbanSound8K). Submit with:
#   sbatch --job-name=xares_hubert --export=MODEL=hubert scripts/slurm/run_xares.sbatch
# Data must already be downloaded (xares_eval/download_zenodo_task.py) --
# this job does NOT attempt X-ARES's own broken bulk-download path.
set -euo pipefail

: "${MODEL:?Set MODEL, e.g. sbatch --export=MODEL=hubert scripts/slurm/run_xares.sbatch}"

module load python/3.11 cuda/12.6 gcc arrow/25.0.0
source "${SCRATCH:-$HOME/scratch}/audio-comp-venv/bin/activate"

export HF_HOME="${SCRATCH:-$HOME/scratch}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export RAYON_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# attr_from_py_path() (inside xares.run) does a real import_module() on
# the encoder/task path with "/" replaced by "." — it only works for
# paths relative to CWD that resolve to a valid nested module name, so
# CWD must be the repo root (not $SCRATCH; env_root is redirected there
# via xares_eval/tasks_scratch/ instead — see those files' docstrings).
cd "$HOME/audio_comp"
export PYTHONPATH="$HOME/audio_comp:${PYTHONPATH:-}"
export XARES_ENV_ROOT="${SCRATCH:-$HOME/scratch}/audio_comp/xares_env/env"
mkdir -p "$XARES_ENV_ROOT"

python -m xares.run --max-jobs 1 \
    xares_eval/encoders/${MODEL}_encoder.py \
    xares_eval/tasks_scratch/freemusicarchive_genre_task.py \
    xares_eval/tasks_scratch/librispeech_asr_task.py \
    xares_eval/tasks_scratch/urbansound8k_task.py
