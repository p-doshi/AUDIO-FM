"""Frozen-embedding counterpart to stage5_lora_finetune_mimii.py's LoRA
run, under the exact same leave-one-machine-type-out split and
class-balanced subsample -- built specifically to disambiguate the LoRA
run's near-chance result (2026-08-16, mean accuracy 0.458-0.499 across 5
models): is machine-type generalization just a hard task for MIMII
anomaly detection regardless of adaptation, or is it specifically a LoRA
adaptation failure?

Unlike the LoRA run, this isn't restricted to LoRA-compatible
architectures -- freezing the backbone entirely means no LoRA/PEFT
involvement, so it runs against any model in audio_comp.models (all 19
active models), same as frozen_baseline_streaming.py's own logic for the
confidential vessel data. Embeddings are extracted once per fold (not
per epoch) since the backbone is frozen -- much cheaper than the LoRA
run, no repeated forward passes through the encoder during training.

Usage:
    python -m audio_comp.pipelines.stage5_frozen_finetune_mimii --model wav2vec2
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from audio_comp.models import get_model_class
from audio_comp.pipelines.stage5_lora_finetune_mimii import (
    MACHINE_TYPES,
    NUM_FOLDS,
    _read_mono,
    build_clip_index,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
NUM_CLASSES = 2
BATCH_SIZE = 16
EPOCHS = 30
EARLY_STOP_PATIENCE = 5
LEARNING_RATE = 1e-3
SEED = 0


def embed_all_clips(adapter, clips: list[dict]) -> np.ndarray:
    embeddings = []
    for start in range(0, len(clips), BATCH_SIZE):
        batch = clips[start : start + BATCH_SIZE]
        waveforms = [_read_mono(c["path"]) for c in batch]
        batch_embeds = adapter.embed_batch(waveforms, 16000)
        embeddings.append(batch_embeds)
        if adapter.device == "cuda":
            torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)


def run_fold(model_name: str, test_machine: str, device: str) -> float:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    clips = build_clip_index()
    train_clips = [c for c in clips if c["machine"] != test_machine]
    test_clips = [c for c in clips if c["machine"] == test_machine]

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
    for test_machine in MACHINE_TYPES:
        acc = run_fold(model_name, test_machine, device)
        print(f"[frozen:{model_name}] held-out machine={test_machine} test accuracy: {acc:.4f}", flush=True)
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    print(f"[frozen:{model_name}] mean accuracy across {NUM_FOLDS} leave-one-machine-out folds: {mean_acc:.4f}")

    out_csv = REPO_ROOT / "results" / "stage5_frozen_mimii.csv"
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

    import yaml

    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        active_models = sorted(yaml.safe_load(f)["active_models"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=active_models)
    args = parser.parse_args()
    main(args.model)
