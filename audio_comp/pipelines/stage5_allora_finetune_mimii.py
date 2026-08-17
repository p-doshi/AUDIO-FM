"""ALLoRA counterpart to stage5_lora_finetune_mimii.py -- identical
protocol (leave-one-machine-type-out, 4 folds, ~200 balanced clips/
class/machine, same epoch count) except the adaptation method, per the
user's explicit request to run all fine-tuning experiments with ALLoRA
after the 7-candidate-method research pass (see journal.md 2026-08-17
and allora.py's module docstring for why ALLoRA was chosen).

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
from audio_comp.pipelines.stage5_lora_finetune_mimii import (
    EPOCHS,
    LEARNING_RATE,
    BATCH_SIZE,
    SEED,
    MIMII_NATIVE_SAMPLE_RATE,
    NUM_CLASSES,
    MACHINE_TYPES,
    NUM_FOLDS,
    build_clip_index,
    _read_mono,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


def run_fold(model_name: str, test_machine: str, device: str) -> float:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    clips = build_clip_index()
    train_clips = [c for c in clips if c["machine"] != test_machine]
    test_clips = [c for c in clips if c["machine"] == test_machine]

    trainable, forward_fn, _, embed_dim, adapter = build_allora_model_and_head(
        model_name, device, NUM_CLASSES, get_model_class
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
    with torch.no_grad():
        for start in range(0, len(test_clips), BATCH_SIZE):
            batch = test_clips[start : start + BATCH_SIZE]
            waveforms = [_read_mono(c["path"]) for c in batch]
            labels = torch.tensor([c["label"] for c in batch], device=device)
            inputs = allora_prepare_inputs(model_name, adapter, waveforms, MIMII_NATIVE_SAMPLE_RATE, device)
            logits = forward_fn(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(batch)

    return correct / total


def main(model_name: str) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model '{model_name}' not in ALLoRA-verified set: {SUPPORTED_MODELS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_accs = []
    for test_machine in MACHINE_TYPES:
        acc = run_fold(model_name, test_machine, device)
        print(f"[allora:{model_name}] held-out machine={test_machine} test accuracy: {acc:.4f}")
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    print(f"[allora:{model_name}] mean accuracy across {NUM_FOLDS} leave-one-machine-out folds: {mean_acc:.4f}")

    out_csv = REPO_ROOT / "results" / "stage5_allora_mimii.csv"
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
