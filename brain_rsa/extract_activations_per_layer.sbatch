#!/bin/bash
#SBATCH --account=def-spadon_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:30:00
#SBATCH --job-name=brain_rsa_perlayer
#SBATCH --output=/scratch/pdoshi/audio_comp/slurm_logs/%x-%j.out
#
# Per-layer extraction for the brain_rsa objective-vs-domain disambiguation
# (mert, clap added 2026-08-13 per a peer-session methodology critique, see
# extract_activations_per_layer.py's docstring and journal.md). Submit with:
#   sbatch --export=MODELS="mert clap" brain_rsa/extract_activations_per_layer.sbatch
set -euo pipefail

: "${MODELS:?Set MODELS, e.g. sbatch --export=MODELS=\"mert clap\" brain_rsa/extract_activations_per_layer.sbatch}"

module load python/3.11 cuda/12.6 gcc arrow/25.0.0
source "${SCRATCH:-$HOME/scratch}/audio-comp-venv/bin/activate"

cd "$HOME/audio_comp"
export PYTHONPATH="$HOME/audio_comp:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH:-$HOME/scratch}/hf_cache"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export RAYON_NUM_THREADS=1

python brain_rsa/extract_activations_per_layer.py --models $MODELS
