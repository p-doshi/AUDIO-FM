"""Stage 5 MIMII extension: matched LoRA fine-tuning on MIMII (industrial
machine anomaly detection), applied to a much larger, genuinely-never-
trained-on OOD domain -- built specifically to test whether Stage 5 v1's
diagnosed problem (overfitting on DeepShip's 63-clip dataset, see
the project notes 2026-08-11 and the project plan's Stage 5 v1 entry) was a real
cross-model adaptability signal drowned by tiny-data noise, or something
else.

**Extended 2026-08-16 from the original 5-model set to all 14 models
covered by `lora_model_configs.py`'s shared LoRA wiring** (the original
5-model DeepShip config plus data2vec_audio/mms/sew/unispeech_sat/wavlm/
whisper/wav2vec2_conformer/bird_mae/clap -- see that module's docstring
for the per-model target-module verification and the 5 models still
excluded: panns_cnn14/audio_jepa/audiomae/musicfm/encodecmae).

**Design choices, stated explicitly (the project plan's own convention):**
- Binary classification (normal vs. abnormal), the dataset's own
  canonical task -- see `xares_eval/tasks_scratch/mimii_task.py`.
- **Leave-one-machine-type-out, 4 folds** (fan/pump/slider/valve), not
  the 16-unit folds `xares_eval/mimii/build_manifest.py` uses for the
  frozen-embedding X-ARES probe. Two reasons: (1) full 16-fold x 10-epoch
  LoRA fine-tuning across 18,019 clips x 5 models is not a bounded
  compute budget: fine-tuning updates encoder weights every step, unlike
  the frozen-embedding X-ARES probe, so it's not comparable cost; (2)
  leave-one-*machine-type*-out is arguably a *harder*, more informative
  generalization test than leave-one-unit-out anyway -- it asks whether
  LoRA-adapted representations generalize to an entirely unseen machine
  type, not just an unseen unit of an already-seen type.
- **Subsampled per machine type** (seeded, `MAX_PER_CLASS_PER_MACHINE =
  200`) to keep the run tractable: pump only has 456 abnormal clips
  total, so 200/class/machine keeps every machine type's per-fold data
  volume in the same ballpark rather than fan (1,475 abnormal) dominating
  train time. This is NOT the full MIMII dataset -- read accuracy
  numbers as being on this bounded, class-balanced subsample, not the
  full 18,019-clip, 4.5:1-imbalanced set the X-ARES probe task uses.
- **EPOCHS=5, not DeepShip's 10** -- MIMII's subsampled training set
  (~1,200 clips/fold) is ~30x DeepShip's (~42 clips/fold); fewer epochs
  needed to reach a comparable number of gradient steps, and the whole
  point of this run is to test whether *more real data* (not more
  epochs) fixes v1's overfitting diagnosis. Matched across all models
  within this run, same as every other hyperparameter -- never blended
  with the DeepShip run's own epoch count as if they were one comparison.
- Same LoRA config as DeepShip (`rank=8, alpha=16`, per-model
  `target_modules`, `dropout=0.05`) via `lora_model_configs.py`'s shared
  per-model wiring rather than a single hardcoded config.

Usage:
    python -m audio_comp.pipelines.stage5_lora_finetune_mimii --model wav2vec2
"""
from __future__ import annotations

import csv
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from audio_comp.models import get_model_class
from audio_comp.pipelines.generic_lora_trainer import write_per_clip_predictions, write_results_idempotent
from audio_comp.pipelines.lora_model_configs import (
    LORA_SUPPORTED_MODELS as SUPPORTED_MODELS,
    build_lora_model_and_head,
    lora_prepare_inputs,
)


def _read_mono(path: str) -> np.ndarray:
    """MIMII wavs are 8-channel mic-array recordings -- downmix to mono,
    matching audio_comp/data/sources/mimii.py's own convention exactly."""
    waveform, _ = sf.read(path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = REPO_ROOT / "data" / "mimii_manifest.csv"
MACHINE_TYPES = ["fan", "pump", "slider", "valve"]
NUM_FOLDS = len(MACHINE_TYPES)
NUM_CLASSES = 2
MAX_PER_CLASS_PER_MACHINE = 300  # raised from 200 (2026-09-03, explicit user request) -- 4 machines x 2 labels x 300 = 2400 total clips, matching the vessel domain's scale (~2463). Verified every one of the 8 (machine, condition) groups in the raw manifest has well over 300 available clips (min 456, pump-abnormal) before raising this, so no group needs padding/reuse to hit the new cap. This constant is the single source of truth for build_clip_index(), imported by all 3 MIMII fine-tuning scripts (frozen/lora/allora) and the geometry loader (rsa_cka_finetuned.py's _load_mimii()) -- one change keeps all of them consistent. Real, flagged consequence: every existing MIMII accuracy result (all conditions, all seeds already run) was computed against the OLD 1600-clip pool and is no longer directly comparable to anything re-run after this change -- a full MIMII re-run is needed, not just new seeds.

# All MIMII wavs are natively 16kHz (verified directly via sf.info(), not
# assumed) -- this is the TRUE sample rate of the raw waveforms read by
# _read_mono(), and must be passed to lora_prepare_inputs() as the
# resample-FROM rate. **Real bug fixed 2026-08-16**: run_fold() used to
# pass build_lora_model_and_head()'s returned `sample_rate` (the MODEL's
# *expected* rate, e.g. 24000 for mert) as if it were this native rate --
# for any model whose expected rate isn't 16000 (mert=24000), that made
# resample() a silent no-op instead of an actual resample, feeding the
# model pitch/speed-corrupted audio. Coincidentally harmless for the
# other 4 original models (wav2vec2/hubert/music2vec/ast all expect
# 16000 too), but a real, confirmed bug for mert and would have silently
# affected any future model whose expected rate differs from 16000.
# Discovered while extending this script to the full 14-model roster --
# same-shaped bug also exists, unfixed, in stage5_lora_finetune.py's
# DeepShip run (32kHz-native clips, every one of its 5 models expects
# 16 or 24kHz) -- left alone there per explicit user direction (DeepShip
# is being deprioritized in favor of the confidential vessel data, not
# worth re-running), but the DeepShip Stage 5 v1 "matched LoRA
# underperforms frozen probing" finding should be read with this
# confound in mind, not purely as an overfitting result.
MIMII_NATIVE_SAMPLE_RATE = 16000

EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
SEED = 0

# 2026-09-07: wav2vec2_conformer's relative-position attention (_apply_relative_embeddings)
# is far more memory-hungry per sample than every other LoRA-covered model -- confirmed via
# a genuine CUDA OOM on a 39.5GB MIG slice at BATCH_SIZE=16 (mimii_lora/allora_wav2vec2_conformer_s0,
# jobs 58036101/58036102, both FAILED 2026-09-06). Same fix already proven elsewhere in this
# project for this exact model (the restricted-domain variant's MODEL_BATCH_SIZE_OVERRIDES).
MODEL_BATCH_SIZE_OVERRIDES = {"wav2vec2_conformer": 4}


CLIP_POOL_SEED = 0  # fixed, NOT the run seed -- which 2400 clips are in the pool at all is now seed-independent (2026-09-03 fix), matching every other domain's convention (fma_genre/urbansound8k/birdclef/librispeech use their full manifest unconditionally; only train/val split + model init vary by seed there). Before this fix, `seed` here reselected which MAX_PER_CLASS_PER_MACHINE clips were even IN the pool, on top of controlling the fold/init randomness downstream -- a real cross-domain inconsistency (MIMII's "seed" changed the underlying data itself, not just how it was used) that the other 4 domains don't have, flagged and fixed per explicit user request rather than left silently.


def build_clip_index(seed: int = SEED) -> list[dict]:
    """`seed` no longer affects WHICH clips are in the pool (see
    CLIP_POOL_SEED above) -- it's accepted here only for backward-
    compatible call signatures; callers that want seed-dependent
    randomness (model init, fold/split order) apply it themselves
    downstream, same pattern as every other domain's loader."""
    with open(MANIFEST_CSV) as f:
        rows = list(csv.DictReader(f))

    rng = np.random.default_rng(CLIP_POOL_SEED)
    by_machine_label: dict[tuple[str, str], list[dict]] = {}
    for row in rows:
        key = (row["machine"], row["condition"])
        by_machine_label.setdefault(key, []).append(row)

    subsampled = []
    for (machine, condition), group in sorted(by_machine_label.items()):
        idx = rng.permutation(len(group))[:MAX_PER_CLASS_PER_MACHINE]
        for i in idx:
            row = group[i]
            subsampled.append(
                dict(path=row["file"], machine=machine, label=int(row["label"]))
            )

    # Pool-provenance fingerprint, printed once per run (2026-09-07). Added after a
    # false-alarm investigation into 28 MIMII result rows that were byte-identical to
    # the archived 1,600-clip-pool backup: with a near-chance binary task on a
    # perfectly balanced test set, a collapsed constant-predicting head scores exactly
    # 0.5 under ANY pool size, and coarse accuracy granularity (k/400 vs k/600) makes
    # incidental exact matches common -- so "result equals the old-pool result" is NOT
    # evidence the old pool was used, and there is no on-disk cache here that could
    # cause that (build_clip_index reads data/mimii_manifest.csv fresh every call).
    # Rather than add a cache key to a pipeline that has no cache, log the pool's
    # actual composition so any future "did this run use the new pool?" question is
    # answered directly from the Slurm log instead of inferred from accuracy values.
    fingerprint = hashlib.sha256(
        "\n".join(sorted(c["path"] for c in subsampled)).encode()
    ).hexdigest()[:12]
    print(
        f"[mimii-pool] n_clips={len(subsampled)} "
        f"max_per_class_per_machine={MAX_PER_CLASS_PER_MACHINE} "
        f"clip_pool_seed={CLIP_POOL_SEED} sha256={fingerprint}",
        flush=True,
    )
    return subsampled


def run_fold(model_name: str, test_machine: str, device: str, seed: int = SEED) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    clips = build_clip_index(seed)
    train_clips = [c for c in clips if c["machine"] != test_machine]
    test_clips = [c for c in clips if c["machine"] == test_machine]

    trainable, forward_fn, sample_rate, embed_dim, adapter = build_lora_model_and_head(
        model_name, device, NUM_CLASSES, get_model_class
    )
    optimizer = torch.optim.AdamW([p for p in trainable.parameters() if p.requires_grad], lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    batch_size = MODEL_BATCH_SIZE_OVERRIDES.get(model_name, BATCH_SIZE)
    rng = np.random.default_rng(seed)
    trainable.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(train_clips))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            batch = [train_clips[i] for i in batch_idx]
            waveforms = [_read_mono(c["path"]) for c in batch]
            labels = torch.tensor([c["label"] for c in batch], device=device)

            inputs = lora_prepare_inputs(model_name, adapter, waveforms, MIMII_NATIVE_SAMPLE_RATE, device)
            logits = forward_fn(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(
            f"  [{model_name} test_machine={test_machine}] epoch {epoch + 1}/{EPOCHS} "
            f"loss={total_loss / len(train_clips):.4f}",
            flush=True,
        )

    trainable.eval()
    correct, total = 0, 0
    predictions_records = []
    with torch.no_grad():
        for start in range(0, len(test_clips), batch_size):
            batch = test_clips[start : start + batch_size]
            waveforms = [_read_mono(c["path"]) for c in batch]
            labels = torch.tensor([c["label"] for c in batch], device=device)
            inputs = lora_prepare_inputs(model_name, adapter, waveforms, MIMII_NATIVE_SAMPLE_RATE, device)
            logits = forward_fn(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(batch)
            preds_cpu, labels_cpu = preds.detach().cpu().tolist(), labels.detach().cpu().tolist()
            for clip, true_label, pred_label in zip(batch, labels_cpu, preds_cpu):
                predictions_records.append({
                    "clip_path": clip["path"], "true_label": true_label,
                    "pred_label": pred_label, "correct": int(true_label == pred_label),
                })

    predictions_out_csv = REPO_ROOT / "results" / "per_clip_predictions" / f"mimii_{model_name}_lora_machine{test_machine}_seed{seed}.csv"
    write_per_clip_predictions(predictions_out_csv, predictions_records)

    return correct / total


def main(model_name: str, seed: int = SEED) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model '{model_name}' not in Stage 5's LoRA-verified set: {SUPPORTED_MODELS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_accs = []
    for test_machine in MACHINE_TYPES:
        acc = run_fold(model_name, test_machine, device, seed)
        print(f"[{model_name}:seed{seed}] held-out machine={test_machine} test accuracy: {acc:.4f}")
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    print(f"[{model_name}:seed{seed}] mean accuracy across {NUM_FOLDS} leave-one-machine-out folds: {mean_acc:.4f}")

    out_csv = REPO_ROOT / "results" / "stage5_lora_mimii.csv"
    new_rows = [[model_name, seed, machine, acc] for machine, acc in zip(MACHINE_TYPES, fold_accs)]
    new_rows.append([model_name, seed, "mean", mean_acc])
    write_results_idempotent(
        out_csv, ["model", "seed", "held_out_machine", "accuracy"], ["model", "seed", "held_out_machine"], new_rows
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for data subsampling + model init/training-order shuffling -- added 2026-08-20 for multi-seed variance runs. Seed 0 matches every prior result already on record.")
    args = parser.parse_args()
    main(args.model, args.seed)
