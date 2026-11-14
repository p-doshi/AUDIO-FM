"""ALLoRA counterpart to stage5_lora_finetune_mimii.py -- identical
protocol (leave-one-machine-type-out, 4 folds, ~200 balanced clips/
class/machine, same epoch count) except the adaptation method, per the
user's explicit request to run all fine-tuning experiments with ALLoRA
after the 7-candidate-method research pass (see the project notes).

Usage:
    python -m audio_comp.pipelines.stage5_allora_finetune_mimii --model wav2vec2
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from audio_comp.models import get_model_class
from audio_comp.pipelines.allora_model_configs import (
    ALLORA_SUPPORTED_MODELS as SUPPORTED_MODELS,
    build_allora_model_and_head,
    allora_prepare_inputs,
)
from audio_comp.pipelines.generic_lora_trainer import write_per_clip_predictions, write_results_idempotent
from audio_comp.pipelines.stage5_lora_finetune_mimii import (
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    MODEL_BATCH_SIZE_OVERRIDES,
    SEED,
    MIMII_NATIVE_SAMPLE_RATE,
    NUM_CLASSES,
    MACHINE_TYPES,
    NUM_FOLDS,
    build_clip_index,
    _read_mono,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_fold(model_name: str, test_machine: str, device: str, seed: int = SEED) -> float:
    torch.manual_seed(seed)
    np.random.seed(seed)

    clips = build_clip_index(seed)
    train_clips = [c for c in clips if c["machine"] != test_machine]
    test_clips = [c for c in clips if c["machine"] == test_machine]

    trainable, forward_fn, _, embed_dim, adapter = build_allora_model_and_head(
        model_name, device, NUM_CLASSES, get_model_class
    )
    optimizer = torch.optim.AdamW([p for p in trainable.parameters() if p.requires_grad], lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    batch_size = MODEL_BATCH_SIZE_OVERRIDES.get(model_name, BATCH_SIZE)
    rng = np.random.default_rng(SEED)
    trainable.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(train_clips))
        total_loss = 0.0
        for start in range(0, len(order), batch_size):
            batch_idx = order[start : start + batch_size]
            batch = [train_clips[i] for i in batch_idx]
            waveforms = [_read_mono(c["path"]) for c in batch]
            labels = torch.tensor([c["label"] for c in batch], device=device)

            inputs = allora_prepare_inputs(model_name, adapter, waveforms, MIMII_NATIVE_SAMPLE_RATE, device)
            logits = forward_fn(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(
            f"  [allora:{model_name} test_machine={test_machine}] epoch {epoch + 1}/{EPOCHS} "
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
            inputs = allora_prepare_inputs(model_name, adapter, waveforms, MIMII_NATIVE_SAMPLE_RATE, device)
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

    predictions_out_csv = REPO_ROOT / "results" / "per_clip_predictions" / f"mimii_{model_name}_allora_machine{test_machine}_seed{seed}.csv"
    write_per_clip_predictions(predictions_out_csv, predictions_records)

    return correct / total


def main(model_name: str, seed: int = SEED) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model '{model_name}' not in ALLoRA-verified set: {SUPPORTED_MODELS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_accs = []
    for test_machine in MACHINE_TYPES:
        acc = run_fold(model_name, test_machine, device, seed)
        print(f"[allora:{model_name}:seed{seed}] held-out machine={test_machine} test accuracy: {acc:.4f}")
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    print(f"[allora:{model_name}:seed{seed}] mean accuracy across {NUM_FOLDS} leave-one-machine-out folds: {mean_acc:.4f}")

    out_csv = REPO_ROOT / "results" / "stage5_allora_mimii.csv"
    new_rows = [[model_name, seed, machine, acc] for machine, acc in zip(MACHINE_TYPES, fold_accs)]
    new_rows.append([model_name, seed, "mean", mean_acc])
    write_results_idempotent(
        out_csv, ["model", "seed", "held_out_machine", "accuracy"], ["model", "seed", "held_out_machine"], new_rows
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for data subsampling + model init -- added 2026-08-20 for multi-seed variance runs. Seed 0 matches every prior result already on record.")
    args = parser.parse_args()
    main(args.model, args.seed)
