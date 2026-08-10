#!/bin/bash
#SBATCH --account=def-spadon_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --job-name=xares_birdclef
#SBATCH --output=/scratch/pdoshi/audio_comp/slurm_logs/%x-%j.out
#
# BirdCLEF 50-species task for one model. Submit with:
#   sbatch --job-name=xares_birdclef_hubert --export=MODEL=hubert scripts/slurm/run_xares_birdclef.sbatch
#
# Data must already be packaged -- see xares_eval/birdclef/build_manifest.py
# and xares_eval/birdclef/make_audio_tar.py (run once, not per model).
set -euo pipefail

: "${MODEL:?Set MODEL, e.g. sbatch --export=MODEL=hubert scripts/slurm/run_xares_birdclef.sbatch}"

module load python/3.11 cuda/12.6 gcc arrow/25.0.0
source "${SCRATCH:-$HOME/scratch}/audio-comp-venv/bin/activate"

export HF_HOME="${SCRATCH:-$HOME/scratch}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export RAYON_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

cd "$HOME/audio_comp"
export PYTHONPATH="$HOME/audio_comp:${PYTHONPATH:-}"
export XARES_ENV_ROOT="${SCRATCH:-$HOME/scratch}/audio_comp/xares_env/env"

python -m xares.run --max-jobs 1 \
    xares_eval/encoders/${MODEL}_encoder.py \
    xares_eval/tasks_scratch/birdclef_task.py
