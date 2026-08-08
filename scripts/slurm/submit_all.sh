#!/bin/bash
# Submits one embedding-extraction Slurm job per model in
# configs/models.yaml's active_models list — this is the "distributed
# across GPUs" piece: N models -> N concurrent jobs, each on its own GPU.
#
# Once the probe set scales past pilot size, extract_embeddings.py could
# additionally be sharded by clip range via $SLURM_ARRAY_TASK_ID for
# data-parallelism *within* a model too — not needed yet at 100 clips.
set -euo pipefail

cd "$(dirname "$0")/../.."

mkdir -p /scratch/pdoshi/audio_comp/slurm_logs

# active_models is a simple flat YAML list; parsed with awk rather than a
# YAML library so this script has no dependency before the venv exists.
MODELS=$(awk '/^active_models:/{flag=1; next} /^deferred_models:/{flag=0} flag && /^  - /{print $2}' configs/models.yaml)

for MODEL in $MODELS; do
    echo "submitting extraction job for $MODEL"
    sbatch --job-name="extract_${MODEL}" --export=MODEL="$MODEL" scripts/slurm/extract_embeddings.sbatch
done
