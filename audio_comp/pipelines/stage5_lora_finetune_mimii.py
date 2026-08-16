"""Stage 5 MIMII extension: matched LoRA fine-tuning on MIMII (industrial
machine anomaly detection), the same 5-model architecture-verified LoRA
policy as `stage5_lora_finetune.py`'s DeepShip run, applied to a much
larger, genuinely-never-trained-on OOD domain -- built specifically to
test whether Stage 5 v1's diagnosed problem (overfitting on DeepShip's
63-clip dataset, see journal.md 2026-08-11 and CLAUDE.md's Stage 5 v1
entry) was a real cross-model adaptability signal drowned by tiny-data
noise, or something else.

**Design choices, stated explicitly (CLAUDE.md's own convention):**
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
  epochs) fixes v1's overfitting diagnosis. Matched across all 5 models
  within this run, same as every other hyperparameter -- never blended
  with the DeepShip run's own epoch count as if they were one comparison.
- Same LoRA config as DeepShip (`rank=8, alpha=16,
  target_modules=["q_proj","v_proj"], dropout=0.05`) and the same
  `SUPPORTED_MODELS` list, for the same reason DeepShip's run was scoped
  there: these 5 models were the ones with directly-verified matching
  attention submodule names (see stage5_lora_finetune.py's docstring).

Usage:
    python -m audio_comp.pipelines.stage5_lora_finetune_mimii --model wav2vec2
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from audio_comp.pipelines.stage5_lora_finetune import (
    SUPPORTED_MODELS,
    build_model_and_head,
    prepare_inputs,
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
MAX_PER_CLASS_PER_MACHINE = 200

EPOCHS = 5
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
SEED = 0


def build_clip_index() -> list[dict]:
    with open(MANIFEST_CSV) as f:
        rows = list(csv.DictReader(f))

    rng = np.random.default_rng(SEED)
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
    return subsampled


def run_fold(model_name: str, test_machine: str, device: str) -> float:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    clips = build_clip_index()
    train_clips = [c for c in clips if c["machine"] != test_machine]
    test_clips = [c for c in clips if c["machine"] == test_machine]

    trainable, forward_fn, sample_rate, embed_dim, adapter = build_model_and_head(
        model_name, device, num_classes=NUM_CLASSES
    )
    optimizer = torch.optim.AdamW([p for p in trainable.parameters() if p.requires_grad], lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    rng = np.random.default_rng(SEED)
    trainable.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(train_clips))
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch_idx = order[start : start + BATCH_SIZE]
            batch = [train_clips[i] for i in batch_idx]
            waveforms = [_read_mono(c["path"]) for c in batch]
            labels = torch.tensor([c["label"] for c in batch], device=device)

            inputs = prepare_inputs(adapter, model_name, waveforms, sample_rate, device)
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
    with torch.no_grad():
        for start in range(0, len(test_clips), BATCH_SIZE):
            batch = test_clips[start : start + BATCH_SIZE]
            waveforms = [_read_mono(c["path"]) for c in batch]
            labels = torch.tensor([c["label"] for c in batch], device=device)
            inputs = prepare_inputs(adapter, model_name, waveforms, sample_rate, device)
            logits = forward_fn(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(batch)

    return correct / total


def main(model_name: str) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model '{model_name}' not in Stage 5's LoRA-verified set: {SUPPORTED_MODELS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_accs = []
    for test_machine in MACHINE_TYPES:
        acc = run_fold(model_name, test_machine, device)
        print(f"[{model_name}] held-out machine={test_machine} test accuracy: {acc:.4f}")
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    print(f"[{model_name}] mean accuracy across {NUM_FOLDS} leave-one-machine-out folds: {mean_acc:.4f}")

    out_csv = REPO_ROOT / "results" / "stage5_lora_mimii.csv"
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "held_out_machine", "accuracy"])
        for machine, acc in zip(MACHINE_TYPES, fold_accs):
            writer.writerow([model_name, machine, acc])
        writer.writerow([model_name, "mean", mean_acc])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    args = parser.parse_args()
    main(args.model)
