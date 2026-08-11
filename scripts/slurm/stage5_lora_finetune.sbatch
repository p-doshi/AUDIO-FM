#!/bin/bash
#SBATCH --account=def-spadon_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_2g.20gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --job-name=stage5_lora
#SBATCH --output=/scratch/pdoshi/audio_comp/slurm_logs/%x-%j.out
#
# Stage 5 v1 LoRA fine-tuning on DeepShip. Submit with:
#   sbatch --job-name=stage5_lora_wav2vec2 --export=MODEL=wav2vec2 scripts/slurm/stage5_lora_finetune.sbatch
set -euo pipefail

: "${MODEL:?Set MODEL, e.g. sbatch --export=MODEL=wav2vec2 scripts/slurm/stage5_lora_finetune.sbatch}"

module load python/3.11 cuda/12.6 gcc arrow/25.0.0
source "${SCRATCH:-$HOME/scratch}/audio-comp-venv/bin/activate"

cd "$HOME/audio_comp"
export PYTHONPATH="$HOME/audio_comp:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH:-$HOME/scratch}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export RAYON_NUM_THREADS=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m audio_comp.pipelines.stage5_lora_finetune --model "$MODEL"
