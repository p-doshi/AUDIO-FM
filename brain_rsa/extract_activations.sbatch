#!/bin/bash
#SBATCH --account=def-spadon_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --job-name=brain_rsa_extract
#SBATCH --output=/scratch/pdoshi/audio_comp/slurm_logs/%x-%j.out
#
# One-model final-layer embedding extraction on the Tuckute/Feather/
# McDermott 165-natural-sound stimulus set (brain_rsa/ detour, see
# brain_rsa/README.md). Submit with:
#   sbatch --export=MODEL=wav2vec2 --job-name=brain_rsa_wav2vec2 brain_rsa/extract_activations.sbatch
set -euo pipefail

: "${MODEL:?Set MODEL, e.g. sbatch --export=MODEL=wav2vec2 brain_rsa/extract_activations.sbatch}"

module load python/3.11 cuda/12.6 gcc arrow/25.0.0
source "${SCRATCH:-$HOME/scratch}/audio-comp-venv/bin/activate"

cd "$HOME/audio_comp"
export PYTHONPATH="$HOME/audio_comp:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH:-$HOME/scratch}/hf_cache"

export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export RAYON_NUM_THREADS=1

python brain_rsa/extract_activations.py --models "$MODEL"
