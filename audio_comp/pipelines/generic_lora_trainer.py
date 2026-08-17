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

import audioread.ffdec
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from audio_comp.models import get_model_class
from audio_comp.pipelines.lora_model_configs import build_lora_model_and_head, lora_prepare_inputs
from audio_comp.pipelines.allora_model_configs import build_allora_model_and_head, allora_prepare_inputs

EPOCHS = 8
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EARLY_STOP_PATIENCE = 3
SEED = 0


def _read_mp3_via_ffmpeg(path: str) -> np.ndarray:
    """FMA-small ships .mp3 files, which libsndfile can't decode
    (confirmed: real LibsndfileError). librosa.load()'s automatic
    audioread dispatch was tried next and ALSO failed with
    NoBackendError, despite `audioread.available_backends()` listing
    FFmpegAudioFile as available AND a direct, manual
    `audioread.ffdec.FFmpegAudioFile(path)` call succeeding cleanly in
    isolation -- a real flakiness in audioread's own automatic backend
    dispatch/probing logic, not an environment problem (confirmed via a
    dedicated debug job before reaching for this workaround). Using the
    working backend directly, bypassing librosa/audioread's dispatcher
    entirely, rather than continuing to chase why the automatic path
    fails when the manual path doesn't."""
    with audioread.ffdec.FFmpegAudioFile(path) as f:
        sr = f.samplerate
        channels = f.channels
        buf = b"".join(f)
    pcm = np.frombuffer(buf, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        pcm = pcm.reshape(-1, channels).mean(axis=1)
    return pcm


def read_mono(path: str) -> np.ndarray:
    if path.lower().endswith(".mp3"):
        return _read_mp3_via_ffmpeg(path)
    waveform, _ = sf.read(path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform


def read_batch_skip_bad(clips: list[dict]) -> tuple[list[np.ndarray], list[dict]]:
    """FMA-small ships a handful of genuinely corrupted mp3s (documented
    upstream, e.g. 099134.mp3 -- audio_comp/data/sources/fma.py's own
    probe-set loader already skips these the same way). Returns waveforms
    alongside the SURVIVING clip dicts (not the original `clips` list) so
    labels stay aligned with whichever waveforms actually decoded."""
    waveforms, kept_clips = [], []
    for c in clips:
        try:
            waveforms.append(read_mono(c["path"]))
            kept_clips.append(c)
        except Exception as e:
            print(f"  WARNING: skipping undecodable clip {c['path']}: {repr(e)[:150]}", flush=True)
    return waveforms, kept_clips


def _train_and_eval_adapter(
    method_tag: str,
    build_fn: Callable,
    prepare_inputs_fn: Callable,
    model_name: str,
    train_clips: list[dict],
    val_clips: list[dict],
    test_clips: list[dict],
    num_classes: int,
    native_sample_rate: int,
    device: str,
    epochs: int,
) -> float:
    """Shared core for both train_and_eval_lora() and
    train_and_eval_allora() -- identical training loop, only the model
    builder and input-preparation functions differ (LoRA via peft vs.
    ALLoRA via manual module replacement, see allora_model_configs.py)."""
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    trainable, forward_fn, _, embed_dim, adapter = build_fn(model_name, device, num_classes, get_model_class)
    optimizer = torch.optim.AdamW([p for p in trainable.parameters() if p.requires_grad], lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    def run_epoch_eval(clips: list[dict]) -> float:
        trainable.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, len(clips), BATCH_SIZE):
                batch = clips[start : start + BATCH_SIZE]
                waveforms, batch = read_batch_skip_bad(batch)
                if not waveforms:
                    continue
                labels = torch.tensor([c["label"] for c in batch], device=device)
                inputs = prepare_inputs_fn(model_name, adapter, waveforms, native_sample_rate, device)
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
            waveforms, batch = read_batch_skip_bad(batch)
            if not waveforms:
                continue
            labels = torch.tensor([c["label"] for c in batch], device=device)
            inputs = prepare_inputs_fn(model_name, adapter, waveforms, native_sample_rate, device)
            logits = forward_fn(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        val_acc = run_epoch_eval(val_clips)
        print(f"  [{method_tag}:{model_name}] epoch {epoch + 1}/{epochs} loss={total_loss / len(train_clips):.4f} val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in trainable.state_dict().items() if k in trainable_param_names
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"  [{method_tag}:{model_name}] early stopping at epoch {epoch + 1} (best val_acc={best_val_acc:.4f})")
                break

    trainable.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)
    return run_epoch_eval(test_clips)


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
    return _train_and_eval_adapter(
        "lora", build_lora_model_and_head, lora_prepare_inputs,
        model_name, train_clips, val_clips, test_clips, num_classes, native_sample_rate, device, epochs,
    )


def train_and_eval_allora(
    model_name: str,
    train_clips: list[dict],
    val_clips: list[dict],
    test_clips: list[dict],
    num_classes: int,
    native_sample_rate: int,
    device: str,
    epochs: int = EPOCHS,
) -> float:
    """ALLoRA counterpart to train_and_eval_lora() -- same signature, same
    training loop, only the adapter method differs."""
    return _train_and_eval_adapter(
        "allora", build_allora_model_and_head, allora_prepare_inputs,
        model_name, train_clips, val_clips, test_clips, num_classes, native_sample_rate, device, epochs,
    )


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

    def embed(clips: list[dict]) -> tuple[np.ndarray, list[dict]]:
        embeddings, kept = [], []
        for start in range(0, len(clips), batch_size):
            batch = clips[start : start + batch_size]
            waveforms, batch = read_batch_skip_bad(batch)
            if not waveforms:
                continue
            batch_embeds = adapter.embed_batch(waveforms, adapter.info.expected_sample_rate)
            embeddings.append(batch_embeds)
            kept.extend(batch)
            if device == "cuda":
                torch.cuda.empty_cache()
        return np.concatenate(embeddings, axis=0), kept

    train_emb_np, train_clips = embed(train_clips)
    val_emb_np, val_clips = embed(val_clips)
    test_emb_np, test_clips = embed(test_clips)
    train_emb = torch.tensor(train_emb_np, dtype=torch.float32)
    val_emb = torch.tensor(val_emb_np, dtype=torch.float32)
    test_emb = torch.tensor(test_emb_np, dtype=torch.float32)
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
