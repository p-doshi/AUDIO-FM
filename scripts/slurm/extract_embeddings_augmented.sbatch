#!/bin/bash
#SBATCH --account=def-spadon_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=audio_comp_extract_augmented
#SBATCH --output=/scratch/pdoshi/audio_comp/slurm_logs/%x-%j.out
#
# Same as extract_embeddings.sbatch but for the 1000-clip (500 original +
# 500 pitch-shifted-augmented) subset built by
# audio_comp/data/build_augmented_probe_subset.py -- for the true
# instance-level paired alignment metric (2026-08-10 journal entry: the
# same-category proxy turned out degenerate, perfectly rank-correlated
# with uniformity). Submit with:
#   sbatch --export=MODEL=clap --job-name=extract_aug_clap scripts/slurm/extract_embeddings_augmented.sbatch
set -euo pipefail

: "${MODEL:?Set MODEL, e.g. sbatch --export=MODEL=clap scripts/slurm/extract_embeddings_augmented.sbatch}"

module load python/3.11 cuda/12.6 gcc arrow/25.0.0
source "${SCRATCH:-$HOME/scratch}/audio-comp-venv/bin/activate"

cd "$HOME/audio_comp"
export PYTHONPATH="$HOME/audio_comp:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH:-$HOME/scratch}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export RAYON_NUM_THREADS=1

OUT_DIR="${SCRATCH:-$HOME/scratch}/audio_comp/embeddings_augmented"
mkdir -p "$OUT_DIR"

python -m audio_comp.pipelines.extract_embeddings \
    --model "$MODEL" \
    --manifest data/augmented_pair_extraction_manifest.csv \
    --data-root /scratch/pdoshi/audio_comp/augmented_probe_subset \
    --out "$OUT_DIR/$MODEL.npz"
