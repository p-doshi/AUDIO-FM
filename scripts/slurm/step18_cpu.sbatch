#!/bin/bash
#SBATCH --account=def-PI
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=phys_step18
#SBATCH --output=/scratch/user/audio_comp/slurm_logs/%x-%j.out
set -euo pipefail

module load python/3.11 cuda/12.6 gcc arrow/25.0.0
source "${SCRATCH:-$HOME/scratch}/audio-comp-venv/bin/activate"

cd "$HOME/audio_comp/scripts/physics"
export PYTHONPATH="$HOME/audio_comp:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH:-$HOME/scratch}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export RAYON_NUM_THREADS=1
export OMP_NUM_THREADS=8
export PHYS_ARCH_CACHE="${SCRATCH}/audio_comp/phys_cache/phys_arch_cache.npz"

mkdir -p /scratch/user/audio_comp/slurm_logs
python -u step18_per_model_grid.py
