#!/bin/bash
#SBATCH --account=def-spadon_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=03:00:00
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

# $HOME's quota is small (48GB) and HF checkpoint/dataset downloads add up
# fast — route the cache to $SCRATCH instead of the ~/.cache/huggingface default.
export HF_HOME="${SCRATCH:-$HOME/scratch}/hf_cache"

# hubert/mert loading intermittently deadlocked (futex_do_wait, near-zero
# CPU, non-deterministic — sometimes 3.6s, sometimes the full 1hr timeout)
# — the classic Rust-thread-pool-after-fork hang from tokenizers/hf_xet.
# These env vars serialize/disable those thread pools to avoid the race.
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export RAYON_NUM_THREADS=1

OUT_DIR="${SCRATCH:-$HOME/scratch}/audio_comp/embeddings"
mkdir -p "$OUT_DIR"

python -m audio_comp.pipelines.extract_embeddings \
    --model "$MODEL" \
    --out "$OUT_DIR/$MODEL.npz"
