#!/bin/bash
# For every (domain, cond, seed) job: ensure all 18 per-model rank caches exist
# (each via its own short-lived subprocess, retried individually), then run the
# lightweight correlation-only assembly step.
SP=/scratch/user/audio_comp/appendix_work
LOG="$SP/cross18_orchestrate.log"
OUT="$SP/cross_18x18"

LORA_MODELS="ast bird_mae clap data2vec_audio hubert mert mms music2vec sew unispeech_sat wav2vec2 wav2vec2_conformer wavlm whisper"
TOPK_MODELS="audio_jepa audiomae musicfm panns_cnn14"

ensure_ranked() {
  local dom=$1 cond=$2 model=$3 seed=$4
  local cache="$SP/cross18_rank_cache/${dom}_${cond}_${model}_seed${seed}.npy"
  if [ -f "$cache" ]; then return 0; fi
  local attempt=1
  while [ $attempt -le 6 ]; do
    python3 "$SP/rank_one_model.py" "$dom" "$cond" "$model" "$seed" >> "$LOG" 2>&1
    if [ -f "$cache" ]; then return 0; fi
    echo "$(date +%H:%M:%S) RETRY $dom $cond $model seed$seed (attempt $attempt failed)" >> "$LOG"
    attempt=$((attempt+1))
    sleep 3
  done
  echo "$(date +%H:%M:%S) GAVE UP $dom $cond $model seed$seed" >> "$LOG"
  return 1
}

while IFS=' ' read -r dom cond seed; do
  [ -z "$dom" ] && continue
  outfile="$OUT/${dom}_${cond}_topk_cross_seed${seed}.csv"
  if [ -f "$outfile" ]; then
    echo "$(date +%H:%M:%S) SKIP (exists) $dom $cond $seed" >> "$LOG"
    continue
  fi
  echo "$(date +%H:%M:%S) START $dom $cond $seed" >> "$LOG"
  for m in $LORA_MODELS; do
    ensure_ranked "$dom" "$cond" "$m" "$seed"
  done
  for m in $TOPK_MODELS; do
    ensure_ranked "$dom" "topk" "$m" "$seed"
  done
  # assembly (correlation-only, fast, low memory)
  python3 "$SP/build_18x18_cross.py" "$dom" "$cond" "$seed" >> "$LOG" 2>&1
  echo "$(date +%H:%M:%S) DONE $dom $cond $seed" >> "$LOG"
done < "$SP/joblist_cross18.txt"

echo "$(date +%H:%M:%S) ALL CROSS18 ORCHESTRATION COMPLETE" >> "$LOG"
