#!/bin/bash
#SBATCH --account=def-spadon_gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=audio_comp_extract
#SBATCH --output=/scratch/pdoshi/audio_comp/slurm_logs/%x-%j.out
#
# One-model embedding-extraction job. Submit with:
#   sbatch --export=MODEL=clap --job-name=extract_clap scripts/slurm/extract_embeddings.sbatch
# submit_all.sh loops this over every model in configs/models.yaml's
# active_models list, submitting one job per model in parallel.
set -euo pipefail

: "${MODEL:?Set MODEL, e.g. sbatch --export=MODEL=clap scripts/slurm/extract_embeddings.sbatch}"

module load python/3.11 cuda/12.6 gcc arrow/25.0.0
source "$HOME/audio-comp-venv/bin/activate"

cd "$HOME/audio_comp"

OUT_DIR="${SCRATCH:-$HOME/scratch}/audio_comp/embeddings"
mkdir -p "$OUT_DIR"

python -m audio_comp.pipelines.extract_embeddings \
    --model "$MODEL" \
    --out "$OUT_DIR/$MODEL.npz"
