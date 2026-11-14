#!/bin/bash
#SBATCH --account=def-PI_gpu
#SBATCH --gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --job-name=phys_step16_17
#SBATCH --output=/scratch/user/audio_comp/slurm_logs/%x-%j.out
#
# Step 16 (multi-teacher feature distillation) + Step 17 (residual breakdown).
# Submit:  sbatch scripts/slurm/step16_multiteacher.sbatch
set -euo pipefail

module load python/3.11 cuda/12.6 gcc arrow/25.0.0
source "${SCRATCH:-$HOME/scratch}/audio-comp-venv/bin/activate"

cd "$HOME/audio_comp/scripts/physics"
export PYTHONPATH="$HOME/audio_comp:${PYTHONPATH:-}"
export HF_HOME="${SCRATCH:-$HOME/scratch}/hf_cache"
export TOKENIZERS_PARALLELISM=false
export HF_HUB_DISABLE_XET=1
export RAYON_NUM_THREADS=1
export PHYS_ARCH_CACHE="${SCRATCH}/audio_comp/phys_cache/phys_arch_cache.npz"

mkdir -p /scratch/user/audio_comp/slurm_logs

echo "===== STEP 16 ====="
python -u step16_multiteacher.py || echo "STEP16 FAILED (continuing)"
echo "===== STEP 17 ====="
python -u step17_residual_breakdown.py || echo "STEP17 FAILED"
echo "===== DONE ====="
