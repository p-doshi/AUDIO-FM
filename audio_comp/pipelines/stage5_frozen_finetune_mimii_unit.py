"""Frozen-embedding counterpart to stage5_lora_finetune_mimii_unit.py's
easier within-machine-type leave-one-unit-out split -- same pairing
pattern as stage5_frozen_finetune_mimii.py vs. the machine-type-out LoRA
run. Runs against all 19 active models (no LoRA-compatibility
restriction).

Usage:
    python -m audio_comp.pipelines.stage5_frozen_finetune_mimii_unit --model wav2vec2
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from audio_comp.models import get_model_class
from audio_comp.pipelines.stage5_lora_finetune_mimii_unit import (
    HELD_OUT_UNIT_SUFFIX,
    MACHINE_TYPES,
    build_clip_index_for_type,
)
from audio_comp.pipelines.stage5_frozen_finetune_mimii import embed_all_clips

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 1e-3
SEED = 0


def run_fold(model_name: str, machine: str, device: str) -> float:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    held_out_unit = f"{machine}_{HELD_OUT_UNIT_SUFFIX}"
    clips = build_clip_index_for_type(machine)
    train_clips = [c for c in clips if c["unit"] != held_out_unit]
    test_clips = [c for c in clips if c["unit"] == held_out_unit]

    adapter = get_model_class(model_name)(device=device)
    adapter.load()

    train_emb = embed_all_clips(adapter, train_clips)
    test_emb = embed_all_clips(adapter, test_clips)
    embed_dim = train_emb.shape[1]

    n_val = max(1, int(0.15 * len(train_clips)))
    rng = np.random.default_rng(SEED)
    val_idx = rng.choice(len(train_clips), size=n_val, replace=False)
    val_mask = np.zeros(len(train_clips), dtype=bool)
    val_mask[val_idx] = True

    train_emb_t = torch.tensor(train_emb[~val_mask], dtype=torch.float32)
    val_emb_t = torch.tensor(train_emb[val_mask], dtype=torch.float32)
    test_emb_t = torch.tensor(test_emb, dtype=torch.float32)
    train_labels = torch.tensor([c["label"] for i, c in enumerate(train_clips) if not val_mask[i]], dtype=torch.long)
    val_labels = torch.tensor([c["label"] for i, c in enumerate(train_clips) if val_mask[i]], dtype=torch.long)
    test_labels = torch.tensor([c["label"] for c in test_clips], dtype=torch.long)

    head = nn.Linear(embed_dim, NUM_CLASSES)
    optimizer = torch.optim.Adam(head.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    def eval_acc(emb_t, labels_t):
        head.eval()
        with torch.no_grad():
            preds = head(emb_t).argmax(dim=-1)
        return (preds == labels_t).float().mean().item()

    best_val_acc, best_state, epochs_without_improvement = -1.0, None, 0
    for epoch in range(EPOCHS):
        head.train()
        optimizer.zero_grad()
        loss = criterion(head(train_emb_t), train_labels)
        loss.backward()
        optimizer.step()

        val_acc = eval_acc(val_emb_t, val_labels)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                break

    head.load_state_dict(best_state)
    return eval_acc(test_emb_t, test_labels)


def main(model_name: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_accs = []
    for machine in MACHINE_TYPES:
        acc = run_fold(model_name, machine, device)
        print(f"[frozen:{model_name}] machine={machine} held-out unit test accuracy: {acc:.4f}", flush=True)
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    print(f"[frozen:{model_name}] mean accuracy across {len(MACHINE_TYPES)} within-type leave-one-unit-out folds: {mean_acc:.4f}")

    out_csv = REPO_ROOT / "results" / "stage5_frozen_mimii_unit.csv"
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "machine_type", "accuracy"])
        for machine, acc in zip(MACHINE_TYPES, fold_accs):
            writer.writerow([model_name, machine, acc])
        writer.writerow([model_name, "mean", mean_acc])


if __name__ == "__main__":
    import argparse

    import yaml

    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        active_models = sorted(yaml.safe_load(f)["active_models"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=active_models)
    args = parser.parse_args()
    main(args.model)
