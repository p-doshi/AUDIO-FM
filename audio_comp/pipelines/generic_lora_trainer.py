"""Shared LoRA/frozen matched-fine-tuning training loop, factored out
after writing the same ~80-line loop 4 times across DeepShip/MIMII/
MIMII-unit/vessel scripts with only the data-loading differing. Takes
plain (waveform, label) lists rather than owning any dataset-specific
manifest/clip logic -- each dataset gets a thin wrapper script that
builds those lists and calls into this module.

Not used to retroactively refactor the earlier per-dataset scripts
(stage5_lora_finetune*.py, run_all_vessel_experiments.py) -- they work,
touching them isn't worth the risk/time right now. This is for FMA-genre/
BirdCLEF/UrbanSound8K and any future dataset.
"""
from __future__ import annotations

from typing import Callable

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from audio_comp.models import get_model_class
from audio_comp.pipelines.lora_model_configs import build_lora_model_and_head, lora_prepare_inputs

EPOCHS = 8
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EARLY_STOP_PATIENCE = 3
SEED = 0


def read_mono(path: str) -> np.ndarray:
    waveform, _ = sf.read(path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform


def train_and_eval_lora(
    model_name: str,
    train_clips: list[dict],
    val_clips: list[dict],
    test_clips: list[dict],
    num_classes: int,
    native_sample_rate: int,
    device: str,
    epochs: int = EPOCHS,
) -> float:
    """Each clip dict needs 'path' and 'label' keys. Returns test accuracy
    from the best-val-accuracy epoch (early-stopped)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    trainable, forward_fn, _, embed_dim, adapter = build_lora_model_and_head(
        model_name, device, num_classes, get_model_class
    )
    optimizer = torch.optim.AdamW([p for p in trainable.parameters() if p.requires_grad], lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    def run_epoch_eval(clips: list[dict]) -> float:
        trainable.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, len(clips), BATCH_SIZE):
                batch = clips[start : start + BATCH_SIZE]
                waveforms = [read_mono(c["path"]) for c in batch]
                labels = torch.tensor([c["label"] for c in batch], device=device)
                inputs = lora_prepare_inputs(model_name, adapter, waveforms, native_sample_rate, device)
                logits = forward_fn(inputs)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(batch)
        return correct / total

    rng = np.random.default_rng(SEED)
    best_val_acc, best_state, epochs_without_improvement = -1.0, None, 0
    trainable_param_names = {name for name, p in trainable.named_parameters() if p.requires_grad}

    for epoch in range(epochs):
        trainable.train()
        order = rng.permutation(len(train_clips))
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch_idx = order[start : start + BATCH_SIZE]
            batch = [train_clips[i] for i in batch_idx]
            waveforms = [read_mono(c["path"]) for c in batch]
            labels = torch.tensor([c["label"] for c in batch], device=device)
            inputs = lora_prepare_inputs(model_name, adapter, waveforms, native_sample_rate, device)
            logits = forward_fn(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        val_acc = run_epoch_eval(val_clips)
        print(f"  [{model_name}] epoch {epoch + 1}/{epochs} loss={total_loss / len(train_clips):.4f} val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in trainable.state_dict().items() if k in trainable_param_names
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"  [{model_name}] early stopping at epoch {epoch + 1} (best val_acc={best_val_acc:.4f})")
                break

    trainable.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)
    return run_epoch_eval(test_clips)


def train_and_eval_frozen(
    model_name: str,
    train_clips: list[dict],
    val_clips: list[dict],
    test_clips: list[dict],
    num_classes: int,
    device: str,
    batch_size: int = 16,
    epochs: int = 30,
    early_stop_patience: int = 5,
    learning_rate: float = 1e-3,
) -> float:
    """Frozen-embedding linear probe, matching frozen_baseline_streaming.py's
    own approach (no backbone gradients)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    adapter = get_model_class(model_name)(device=device)
    adapter.load()

    def embed(clips: list[dict]) -> np.ndarray:
        embeddings = []
        for start in range(0, len(clips), batch_size):
            batch = clips[start : start + batch_size]
            waveforms = [read_mono(c["path"]) for c in batch]
            batch_embeds = adapter.embed_batch(waveforms, adapter.info.expected_sample_rate)
            embeddings.append(batch_embeds)
            if device == "cuda":
                torch.cuda.empty_cache()
        return np.concatenate(embeddings, axis=0)

    train_emb = torch.tensor(embed(train_clips), dtype=torch.float32)
    val_emb = torch.tensor(embed(val_clips), dtype=torch.float32)
    test_emb = torch.tensor(embed(test_clips), dtype=torch.float32)
    train_labels = torch.tensor([c["label"] for c in train_clips], dtype=torch.long)
    val_labels = torch.tensor([c["label"] for c in val_clips], dtype=torch.long)
    test_labels = torch.tensor([c["label"] for c in test_clips], dtype=torch.long)

    head = nn.Linear(train_emb.shape[1], num_classes)
    optimizer = torch.optim.Adam(head.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()

    def eval_acc(emb, labels) -> float:
        head.eval()
        with torch.no_grad():
            preds = head(emb).argmax(dim=-1)
        return (preds == labels).float().mean().item()

    best_val_acc, best_state, epochs_without_improvement = -1.0, None, 0
    for epoch in range(epochs):
        head.train()
        optimizer.zero_grad()
        loss = criterion(head(train_emb), train_labels)
        loss.backward()
        optimizer.step()

        val_acc = eval_acc(val_emb, val_labels)
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().clone() for k, v in head.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= early_stop_patience:
                break

    head.load_state_dict(best_state)
    return eval_acc(test_emb, test_labels)
