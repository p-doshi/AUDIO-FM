"""Pairwise RSA/CKA on POST-FINE-TUNING embeddings, for the 5 public
(non-confidential) categories -- FMA-genre, UrbanSound8K, BirdCLEF,
LibriSpeech speaker-ID, MIMII. Mirrors the restricted-domain variant's
methodology and 3-condition split (lora/allora/topk, never blended) but
on locally-available data instead of streamed confidential data, and
covers 5 categories instead of 1.

This project's original standing rule was "geometry is always compared
pretrained, never fine-tuned" -- post-adaptation RSA/CKA was only ever
built as a deliberate extension for the confidential vessel domain
(2026-08-18). Added here 2026-08-20 per explicit user request to extend
that same extension to the public categories: does adaptation pull
independently-trained models' representations closer together or push
them further apart, on data with real signal to learn (unlike vessel,
where this question came back null across the board)?

**Fold-based categories (UrbanSound8K, BirdCLEF, MIMII) use ONE FIXED
representative fold/held-out-unit for training** (UrbanSound8K: fold 1
of [1,4,7,10]; BirdCLEF: fold 0 of 5; MIMII: the first machine type in
MACHINE_TYPES) -- a real, explicit scoping choice, not an oversight.
Unlike the accuracy scripts (which average across all folds since a
single accuracy *number* per fold is cheap to report), a geometry
computation needs ONE trained model to extract ONE embedding space from;
averaging RDMs across folds would blend genuinely different fine-tuned
encoders into a fictional "average" model that never existed. The
embedding SET itself is always the full category manifest (all clips,
not just one fold's test split), matching vessel's own "full clip set"
convention.

**Three conditions, computed and reported separately, never blended**,
same reasoning as the restricted-domain variant:
  - LORA / ALLORA: the 14 LoRA/ALLoRA-compatible models.
  - TOPK: the 4 top-K-unfreeze-only models (panns_cnn14/audio_jepa/
    audiomae/musicfm).
No FROZEN condition here -- that's already covered by the existing
`compare_models.py` -> `results/rsa_matrix.csv` pipeline (seed-
independent, no need to duplicate it here).

**Multi-seed**: lora/allora/topk embeddings genuinely differ per seed
(post-adaptation, unlike frozen). Each seed writes its own, fully
separate N-model x N-model matrix file -- never averaged/combined, per
the same explicit instruction already applied to the vessel-domain and
fine-tuning-accuracy reruns this session.

RDM metric: `correlation` (1 - Pearson correlation across embedding
dims), this project's standard RSA choice throughout (Kriegeskorte et
al. 2008; matches `compare_models.py`'s default and the restricted-domain variant)
-- not literal per-pair cosine similarity, though CKA (computed from
Gram matrices, i.e. dot products) is closely related in spirit. State
this explicitly since "cosine values" was the original phrasing used to
request this script; flag if literal cosine-metric RDMs are wanted
instead (`compute_rdm(..., metric="cosine")` is a one-line swap).

Resume support: each model's RDM+Gram is checkpointed per (category,
condition, model, seed) to CHECKPOINT_DIR, deleted once that
(category, condition, seed)'s final matrix is written -- same pattern as
the restricted-domain variant, cheap here too since fine-tuning + full-set embedding
extraction is the expensive part, not disk I/O.

Usage:
    python -m audio_comp.pipelines.rsa_cka_finetuned --category fma_genre --condition lora --seed 0
    python -m audio_comp.pipelines.rsa_cka_finetuned --category mimii --condition topk --seed 2
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from audio_comp.models import get_model_class
from audio_comp.geometry.rdm import compute_rdm
from audio_comp.geometry.rsa import rsa_score
from audio_comp.pipelines.generic_lora_trainer import SEED as DEFAULT_SEED, read_batch_skip_bad as _read_batch_skip_bad_uncached
from audio_comp.pipelines.lora_model_configs import (
    LORA_SUPPORTED_MODELS,
    MODEL_LORA_SPECS,
    build_lora_model_and_head,
    lora_prepare_inputs,
)
from audio_comp.pipelines.allora_model_configs import (
    ALLORA_SUPPORTED_MODELS,
    build_allora_model_and_head,
    allora_prepare_inputs,
)
from audio_comp.pipelines.topk_unfreeze_configs import (
    TOPK_SUPPORTED_MODELS,
    build_topk_model_and_head,
    topk_prepare_inputs,
    topk_trainable_params,
)
from audio_comp.pipelines._vram import GreedyVRAMClaimer

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
# Overridable via RSA_CKA_CHECKPOINT_DIR (added 2026-09-14) so a one-off
# rerun (e.g. re-extracting a single model after a bug fix) can point at a
# fresh, separately-seeded checkpoint directory instead of either sharing
# the default dir (which would load-and-skip the very model that needs
# re-running, since checkpoints are cached per-model with no fix-aware
# invalidation) or using --fresh (which wipes every model's checkpoint for
# the combo, forcing needless full retrains of models that were already
# correct). Default unchanged for every other caller.
CHECKPOINT_DIR = Path(os.environ.get("RSA_CKA_CHECKPOINT_DIR", "")) if os.environ.get("RSA_CKA_CHECKPOINT_DIR") else REPO_ROOT / "audio_comp" / "pipelines" / ".rsa_finetuned_checkpoints"

# In-memory decoded-waveform cache, added 2026-09-04 after a real
# performance bug was found: read_batch_skip_bad() (generic_lora_trainer.py,
# shared with the accuracy pipeline, not modified here to avoid changing
# its memory footprint for other callers) re-reads and re-decodes every
# clip from disk on every single batch call, with zero caching -- for
# FMA-genre's mp3s specifically that means spawning a fresh ffmpeg
# subprocess per clip, every batch, every epoch. This script calls it for
# every training batch across 8 epochs PLUS a val pass each epoch PLUS
# the final full-clip-set embedding pass, all for the same clips, over
# and over -- with several of these processes running concurrently on
# the secondary cluster competing for the same disk/CPU, this was a real, severe,
# compounding I/O bottleneck (observed: 48 hours to reach epoch 5/8 on
# the very FIRST of 14 models in a job). Caching each clip's decoded (and
# resampled/truncated) waveform after its first read cuts this to one
# real decode per clip per script invocation, regardless of epoch count.
# Safe because target_sample_rate/max_duration_s are constant for the
# whole lifetime of one script invocation (fixed per category, not
# per-model) -- if that ever stops being true this cache would need a
# composite key, flagged here rather than assumed forever.
_waveform_cache: dict[str, tuple] = {}


def read_batch_skip_bad(clips: list[dict], target_sample_rate: int, max_duration_s: float | None = None):
    waveforms, kept_clips = [], []
    to_fetch = [c for c in clips if c["path"] not in _waveform_cache]
    if to_fetch:
        fresh_waveforms, fresh_kept = _read_batch_skip_bad_uncached(to_fetch, target_sample_rate, max_duration_s)
        fresh_kept_paths = {c["path"] for c in fresh_kept}
        for c, w in zip(fresh_kept, fresh_waveforms):
            _waveform_cache[c["path"]] = w
        # Clips that failed to decode get cached as a sentinel (None) too --
        # otherwise a permanently-corrupt clip would trigger a fresh, doomed
        # ffmpeg subprocess attempt on every single batch it appears in,
        # every epoch, the exact redundant-I/O pattern this cache exists to
        # eliminate. FMA-genre's few known-bad mp3s hit this every batch.
        for c in to_fetch:
            if c["path"] not in fresh_kept_paths:
                _waveform_cache[c["path"]] = None
    for c in clips:
        w = _waveform_cache.get(c["path"])
        if w is not None:
            waveforms.append(w)
            kept_clips.append(c)
    return waveforms, kept_clips


EPOCHS = 8
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EARLY_STOP_PATIENCE = 3
# Added 2026-09-07, extending a fix already applied to stage5_lora_finetune_mimii.py/
# stage5_allora_finetune_mimii.py/finetune_data_geometry.py -- wav2vec2_conformer's
# relative-position attention is uniquely memory-hungry and OOMs on a MIG slice at
# the default batch size. Needed here specifically because MIMII's post-adaptation
# geometry jobs (mimii_geom_*) are being moved from full-GPU gres to MIG slices.
MODEL_BATCH_SIZE_OVERRIDES = {"wav2vec2_conformer": 4}


# ---------------------------------------------------------------------------
# Per-category data loading -- reuses each finetune_*.py's own manifest
# logic directly rather than reimplementing it, so this script can never
# silently drift from the accuracy scripts' own train/val construction.
# ---------------------------------------------------------------------------

def _load_fma_genre():
    from audio_comp.pipelines.finetune_fma_genre import load_clips, NUM_CLASSES, NATIVE_SAMPLE_RATE, MAX_CLIP_DURATION_S
    train, val, test = load_clips()
    return train + val + test, train, val, NUM_CLASSES, NATIVE_SAMPLE_RATE, MAX_CLIP_DURATION_S


def _load_librispeech():
    from audio_comp.pipelines.finetune_librispeech import load_clips, NATIVE_SAMPLE_RATE, MAX_CLIP_DURATION_S
    train, val, test, num_classes = load_clips()
    return train + val + test, train, val, num_classes, NATIVE_SAMPLE_RATE, MAX_CLIP_DURATION_S


def _load_urbansound8k():
    from audio_comp.pipelines.finetune_urbansound8k import load_clips, NUM_CLASSES, NATIVE_SAMPLE_RATE, VAL_FRACTION_OF_TRAIN
    REPRESENTATIVE_FOLD = 1  # first of HELD_OUT_FOLDS=[1,4,7,10] -- see module docstring
    clips = load_clips()
    train_pool = [c for c in clips if c["fold"] != REPRESENTATIVE_FOLD]
    rng = np.random.default_rng(DEFAULT_SEED)
    val_idx = set(rng.choice(len(train_pool), size=int(VAL_FRACTION_OF_TRAIN * len(train_pool)), replace=False))
    train = [c for i, c in enumerate(train_pool) if i not in val_idx]
    val = [c for i, c in enumerate(train_pool) if i in val_idx]
    return clips, train, val, NUM_CLASSES, NATIVE_SAMPLE_RATE, None


def _load_birdclef():
    from audio_comp.pipelines.finetune_birdclef import load_clips, NATIVE_SAMPLE_RATE
    REPRESENTATIVE_FOLD = 0  # first of range(NUM_FOLDS=5) -- see module docstring
    clips, label_map = load_clips()
    train_pool = [c for c in clips if c["fold"] != REPRESENTATIVE_FOLD]
    rng = np.random.default_rng(DEFAULT_SEED)
    val_idx = set(rng.choice(len(train_pool), size=max(1, int(0.15 * len(train_pool))), replace=False))
    train = [c for i, c in enumerate(train_pool) if i not in val_idx]
    val = [c for i, c in enumerate(train_pool) if i in val_idx]
    return clips, train, val, len(label_map), NATIVE_SAMPLE_RATE, None


def _load_mimii():
    from audio_comp.pipelines.stage5_lora_finetune_mimii import build_clip_index, MACHINE_TYPES, MIMII_NATIVE_SAMPLE_RATE, NUM_CLASSES
    REPRESENTATIVE_MACHINE = MACHINE_TYPES[0]  # first of the 4 machine types -- see module docstring
    clips = build_clip_index()
    train = [c for c in clips if c["machine"] != REPRESENTATIVE_MACHINE]
    return clips, train, [], NUM_CLASSES, MIMII_NATIVE_SAMPLE_RATE, None


CATEGORY_LOADERS = {
    "fma_genre": _load_fma_genre,
    "librispeech": _load_librispeech,
    "urbansound8k": _load_urbansound8k,
    "birdclef": _load_birdclef,
    "mimii": _load_mimii,
}


# ---------------------------------------------------------------------------
# Checkpointing (RDM + Gram matrix only, per category/condition/model/seed)
# ---------------------------------------------------------------------------

def _checkpoint_path(category: str, condition: str, model_name: str, seed: int) -> Path:
    return CHECKPOINT_DIR / f"{category}_{condition}_{model_name}_seed{seed}.npz"


def _load_checkpoint(category: str, condition: str, model_name: str, seed: int):
    path = _checkpoint_path(category, condition, model_name, seed)
    if not path.exists():
        return None
    data = np.load(path)
    print(f"[{category}:{condition}:{model_name}:seed{seed}] resuming from checkpoint.", flush=True)
    return data["rdm"], data["gram"]


def _save_checkpoint(category: str, condition: str, model_name: str, seed: int, rdm: np.ndarray, gram: np.ndarray) -> None:
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez(_checkpoint_path(category, condition, model_name, seed), rdm=rdm, gram=gram)


def _clear_checkpoints(category: str, condition: str, seed: int) -> None:
    if not CHECKPOINT_DIR.exists():
        return
    for f in CHECKPOINT_DIR.glob(f"{category}_{condition}_*_seed{seed}.npz"):
        f.unlink()


def _rdm_and_gram(embeddings: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return compute_rdm(embeddings), embeddings @ embeddings.T


def _cka_from_grams(gram_x: np.ndarray, gram_y: np.ndarray) -> float:
    n = gram_x.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    cx = centering @ gram_x @ centering
    cy = centering @ gram_y @ centering
    numerator = np.sum(cx * cy)
    denominator = np.sqrt(np.sum(cx * cx) * np.sum(cy * cy))
    return float(numerator / denominator) if denominator != 0 else float("nan")


# ---------------------------------------------------------------------------
# Train (lora/allora/topk) then embed the FULL clip set with the adapted
# encoder -- same shape as the restricted-domain variant's train_lora_and_embed(), just
# sourcing waveforms from local disk (read_batch_skip_bad) instead of a
# streamed ClipCache.
# ---------------------------------------------------------------------------

def _train_and_embed_lora(model_name, all_clips, train_clips, val_clips, num_classes, native_sample_rate, max_duration_s, device, seed) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_size = MODEL_BATCH_SIZE_OVERRIDES.get(model_name, BATCH_SIZE)

    trainable, forward_fn, _, embed_dim, adapter = build_lora_model_and_head(model_name, device, num_classes, get_model_class)
    optimizer = torch.optim.AdamW([p for p in trainable.parameters() if p.requires_grad], lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    trainable_param_names = {name for name, p in trainable.named_parameters() if p.requires_grad}

    def eval_acc(clips):
        trainable.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, len(clips), batch_size):
                batch = clips[start : start + batch_size]
                waveforms, batch = read_batch_skip_bad(batch, native_sample_rate, max_duration_s)
                if not waveforms:
                    continue
                labels = torch.tensor([c["label"] for c in batch], device=device)
                inputs = lora_prepare_inputs(model_name, adapter, waveforms, native_sample_rate, device)
                preds = forward_fn(inputs).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(batch)
        return correct / total if total else 0.0

    rng = np.random.default_rng(seed)
    best_val_acc, best_state, epochs_without_improvement = -1.0, None, 0
    for epoch in range(EPOCHS):
        trainable.train()
        order = rng.permutation(len(train_clips))
        for batch_num, start in enumerate(range(0, len(order), batch_size)):
            batch = [train_clips[i] for i in order[start : start + batch_size]]
            waveforms, batch = read_batch_skip_bad(batch, native_sample_rate, max_duration_s)
            if not waveforms:
                continue
            labels = torch.tensor([c["label"] for c in batch], device=device)
            inputs = lora_prepare_inputs(model_name, adapter, waveforms, native_sample_rate, device)
            loss = criterion(forward_fn(inputs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if device == "cuda" and batch_num % 10 == 0:
                torch.cuda.empty_cache()
        val_acc = eval_acc(val_clips) if val_clips else 0.0
        print(f"[lora-rsa:{model_name}:seed{seed}] epoch {epoch + 1}/{EPOCHS} val_acc={val_acc:.4f}", flush=True)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in trainable.state_dict().items() if k in trainable_param_names}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        trainable.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)
    trainable.eval()

    spec = MODEL_LORA_SPECS[model_name]
    embeddings = []
    with torch.no_grad():
        for batch_num, start in enumerate(range(0, len(all_clips), batch_size)):
            batch = all_clips[start : start + batch_size]
            waveforms, batch = read_batch_skip_bad(batch, native_sample_rate, max_duration_s)
            if not waveforms:
                continue
            inputs = lora_prepare_inputs(model_name, adapter, waveforms, native_sample_rate, device)
            pooled = spec.forward(trainable["peft_model"], inputs)
            embeddings.append(pooled.cpu().numpy())
            # every 10 batches, not every batch (2026-09-04 fix) -- matches
            # the training loop's own cadence; calling this every single
            # batch forces a CUDA sync each time, real avoidable overhead
            # for a no_grad() inference-only pass with much lower memory
            # pressure than training anyway.
            if device == "cuda" and batch_num % 10 == 0:
                torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)


def _train_and_embed_allora(model_name, all_clips, train_clips, val_clips, num_classes, native_sample_rate, max_duration_s, device, seed) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_size = MODEL_BATCH_SIZE_OVERRIDES.get(model_name, BATCH_SIZE)

    trainable, forward_fn, _, embed_dim, adapter = build_allora_model_and_head(model_name, device, num_classes, get_model_class)
    optimizer = torch.optim.AdamW([p for p in trainable.parameters() if p.requires_grad], lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()
    trainable_param_names = {name for name, p in trainable.named_parameters() if p.requires_grad}

    def eval_acc(clips):
        trainable.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, len(clips), batch_size):
                batch = clips[start : start + batch_size]
                waveforms, batch = read_batch_skip_bad(batch, native_sample_rate, max_duration_s)
                if not waveforms:
                    continue
                labels = torch.tensor([c["label"] for c in batch], device=device)
                inputs = allora_prepare_inputs(model_name, adapter, waveforms, native_sample_rate, device)
                preds = forward_fn(inputs).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(batch)
        return correct / total if total else 0.0

    rng = np.random.default_rng(seed)
    best_val_acc, best_state, epochs_without_improvement = -1.0, None, 0
    for epoch in range(EPOCHS):
        trainable.train()
        order = rng.permutation(len(train_clips))
        for batch_num, start in enumerate(range(0, len(order), batch_size)):
            batch = [train_clips[i] for i in order[start : start + batch_size]]
            waveforms, batch = read_batch_skip_bad(batch, native_sample_rate, max_duration_s)
            if not waveforms:
                continue
            labels = torch.tensor([c["label"] for c in batch], device=device)
            inputs = allora_prepare_inputs(model_name, adapter, waveforms, native_sample_rate, device)
            loss = criterion(forward_fn(inputs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if device == "cuda" and batch_num % 10 == 0:
                torch.cuda.empty_cache()
        val_acc = eval_acc(val_clips) if val_clips else 0.0
        print(f"[allora-rsa:{model_name}:seed{seed}] epoch {epoch + 1}/{EPOCHS} val_acc={val_acc:.4f}", flush=True)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {k: v.detach().cpu().clone() for k, v in trainable.state_dict().items() if k in trainable_param_names}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        trainable.load_state_dict({k: v.to(device) for k, v in best_state.items()}, strict=False)
    trainable.eval()

    spec = MODEL_LORA_SPECS[model_name]  # ALLoRA reuses the same per-model forward()/prepare_inputs() spec as LoRA
    embeddings = []
    with torch.no_grad():
        for batch_num, start in enumerate(range(0, len(all_clips), batch_size)):
            batch = all_clips[start : start + batch_size]
            waveforms, batch = read_batch_skip_bad(batch, native_sample_rate, max_duration_s)
            if not waveforms:
                continue
            inputs = allora_prepare_inputs(model_name, adapter, waveforms, native_sample_rate, device)
            pooled = spec.forward(trainable["model"], inputs)
            embeddings.append(pooled.cpu().numpy())
            if device == "cuda" and batch_num % 10 == 0:  # 2026-09-04 fix, see LoRA's identical comment above
                torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)


def _train_and_embed_topk(model_name, all_clips, train_clips, val_clips, num_classes, max_duration_s, device, seed) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    batch_size = MODEL_BATCH_SIZE_OVERRIDES.get(model_name, BATCH_SIZE)

    model, head, forward_fn, sample_rate, embed_dim, adapter = build_topk_model_and_head(model_name, device, num_classes, get_model_class)
    trainable_params = topk_trainable_params(model, head)
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    def eval_acc(clips):
        model.eval()
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, len(clips), batch_size):
                batch = clips[start : start + batch_size]
                waveforms, batch = read_batch_skip_bad(batch, sample_rate, max_duration_s)
                if not waveforms:
                    continue
                labels = torch.tensor([c["label"] for c in batch], device=device)
                inputs = topk_prepare_inputs(model_name, adapter, waveforms, sample_rate, device)
                preds = forward_fn(inputs).argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(batch)
        return correct / total if total else 0.0

    rng = np.random.default_rng(seed)
    best_val_acc, best_state, epochs_without_improvement = -1.0, None, 0
    for epoch in range(EPOCHS):
        model.train()
        head.train()
        order = rng.permutation(len(train_clips))
        for batch_num, start in enumerate(range(0, len(order), batch_size)):
            batch = [train_clips[i] for i in order[start : start + batch_size]]
            waveforms, batch = read_batch_skip_bad(batch, sample_rate, max_duration_s)
            if not waveforms:
                continue
            labels = torch.tensor([c["label"] for c in batch], device=device)
            inputs = topk_prepare_inputs(model_name, adapter, waveforms, sample_rate, device)
            loss = criterion(forward_fn(inputs), labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if device == "cuda" and batch_num % 10 == 0:
                torch.cuda.empty_cache()
        val_acc = eval_acc(val_clips) if val_clips else 0.0
        print(f"[topk-rsa:{model_name}:seed{seed}] epoch {epoch + 1}/{EPOCHS} val_acc={val_acc:.4f}", flush=True)
        if val_acc >= best_val_acc:
            best_val_acc = val_acc
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict({k: v.to(device) for k, v in best_state["model"].items()})
        head.load_state_dict({k: v.to(device) for k, v in best_state["head"].items()})
    model.eval()

    embeddings = []
    with torch.no_grad():
        for batch_num, start in enumerate(range(0, len(all_clips), batch_size)):
            batch = all_clips[start : start + batch_size]
            waveforms, batch = read_batch_skip_bad(batch, sample_rate, max_duration_s)
            if not waveforms:
                continue
            inputs = topk_prepare_inputs(model_name, adapter, waveforms, sample_rate, device)
            # forward_fn(inputs) returns logits (post-head); re-derive the
            # pooled pre-head embedding the same way build_topk_model_and_head's
            # own forward_fn does internally, minus the final head() call.
            from audio_comp.pipelines.topk_unfreeze_configs import TOPK_MODEL_SPECS
            pooled = TOPK_MODEL_SPECS[model_name].forward(model, inputs)
            embeddings.append(pooled.cpu().numpy())
            if device == "cuda" and batch_num % 10 == 0:  # 2026-09-04 fix, see LoRA's identical comment above
                torch.cuda.empty_cache()
    return np.concatenate(embeddings, axis=0)


def write_matrix_csv(matrix: np.ndarray, labels: list[str], path: Path) -> None:
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{v:.4f}" for v in row])


def plot_heatmap(matrix: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1 + len(labels) * 0.5, 1 + len(labels) * 0.5))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="viridis")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90, fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")


def main(category: str, condition: str, seed: int, fresh: bool = False, claim_vram_gb: float | None = None, wait_for_vram_gb: float | None = None, wait_timeout: float | None = None, keep_checkpoints: bool = False) -> None:
    if fresh:
        _clear_checkpoints(category, condition, seed)

    all_clips, train_clips, val_clips, num_classes, native_sample_rate, max_duration_s = CATEGORY_LOADERS[category]()
    print(f"[{category}:{condition}:seed{seed}] {len(all_clips)} total clips, {len(train_clips)} train, {len(val_clips)} val")

    # Pinned to physical GPU 0 explicitly (2026-09-04, same change/reasoning
    # as the restricted-domain scripts/finetune_data_geometry.py/
    # the restricted-domain variant) -- previously just "cuda", relying entirely on
    # whatever launched this (e.g. tca_parallel_runner.py --gpu 0) to set
    # CUDA_VISIBLE_DEVICES correctly. Still works fine through the runner,
    # but this removes the dependency for a standalone/manual invocation.
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # GreedyVRAMClaimer (2026-09-04, per explicit user request) -- without
    # this, a job only grabs memory in one-shot allocations exactly when it
    # needs them, competing fresh against any other process (this project's
    # own or another user's) for whatever's free AT THAT MOMENT. Claiming
    # greedily as soon as this process starts -- and growing opportunistically
    # in the background while data loads -- means memory that frees up while
    # we're waiting gets reserved into THIS process's allocator before a
    # competing process can grab it first, closing exactly the race the
    # orchestrator's own 30s-poll-and-launch loop can't close on its own
    # (it can only launch a NEW process once enough is free; it can't hold
    # a place in line between checks).
    target_gb = wait_for_vram_gb or claim_vram_gb
    if device.startswith("cuda") and target_gb:
        claimer = GreedyVRAMClaimer(device=0, max_gb=target_gb).start()
        if wait_for_vram_gb:
            claimer.wait_until_target(timeout=wait_timeout)
        claimer.release()  # hand the reserved pool to this process's own allocator before any real model load

    models = LORA_SUPPORTED_MODELS if condition == "lora" else (
        ALLORA_SUPPORTED_MODELS if condition == "allora" else TOPK_SUPPORTED_MODELS
    )

    # Heads-up for the results auto-push added 2026-09-07: warn once training
    # starts on the LAST model this combo still needs before it finishes and
    # pushes -- recomputed dynamically each iteration (not a one-time
    # snapshot) so it fires correctly whether resuming a mostly-cached combo
    # or running a fresh one from scratch.
    remaining_uncached = [m for m in models if _load_checkpoint(category, condition, m, seed) is None]

    rdms, grams = {}, {}
    for model_name in models:
        cached = _load_checkpoint(category, condition, model_name, seed)
        if cached is not None:
            rdms[model_name], grams[model_name] = cached
            continue
        if len(remaining_uncached) == 1:
            print(f"[{category}:{condition}:seed{seed}] LAST model ({model_name}) now training -- "
                  f"push to the primary cluster expected once this finishes.", flush=True)
        remaining_uncached.remove(model_name)
        if condition == "lora":
            emb = _train_and_embed_lora(model_name, all_clips, train_clips, val_clips, num_classes, native_sample_rate, max_duration_s, device, seed)
        elif condition == "allora":
            emb = _train_and_embed_allora(model_name, all_clips, train_clips, val_clips, num_classes, native_sample_rate, max_duration_s, device, seed)
        else:
            emb = _train_and_embed_topk(model_name, all_clips, train_clips, val_clips, num_classes, max_duration_s, device, seed)
        rdm, gram = _rdm_and_gram(emb)
        rdms[model_name], grams[model_name] = rdm, gram
        _save_checkpoint(category, condition, model_name, seed, rdm, gram)

    names = sorted(rdms.keys())
    n = len(names)
    rsa_matrix = np.eye(n)
    cka_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rsa_matrix[i, j] = rsa_matrix[j, i] = rsa_score(rdms[names[i]], rdms[names[j]])
            cka_matrix[i, j] = cka_matrix[j, i] = _cka_from_grams(grams[names[i]], grams[names[j]])

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    suffix = f"{category}_{condition}_seed{seed}"
    write_matrix_csv(rsa_matrix, names, RESULTS_DIR / f"rsa_finetuned_matrix_{suffix}.csv")
    write_matrix_csv(cka_matrix, names, RESULTS_DIR / f"cka_finetuned_matrix_{suffix}.csv")
    plot_heatmap(rsa_matrix, names, f"RSA -- {category}, post-{condition}, seed={seed}", RESULTS_DIR / f"rsa_finetuned_heatmap_{suffix}.png")
    plot_heatmap(cka_matrix, names, f"CKA -- {category}, post-{condition}, seed={seed}", RESULTS_DIR / f"cka_finetuned_heatmap_{suffix}.png")
    print(f"Wrote {suffix} RSA/CKA matrices to {RESULTS_DIR}.")
    if keep_checkpoints:
        print(f"[{category}:{condition}:seed{seed}] --keep-checkpoints set -- leaving per-model RDM/gram "
              f"checkpoints on disk (each contains 'rdm'+'gram' keys, needed for any future self-RSA-style "
              f"analysis comparing a model's own frozen vs. adapted RDM -- see scripts/"
              f"extract_fma_genre_frozen_rdm_selfrsa.py's docstring, 2026-09-07, for why this matters).")
    else:
        _clear_checkpoints(category, condition, seed)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--category", choices=list(CATEGORY_LOADERS.keys()), required=True)
    parser.add_argument("--condition", choices=["lora", "allora", "topk"], required=True)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--claim-vram-gb", type=float, default=None, help="Non-blocking: greedily claim whatever GPU memory is free right now (up to this cap) and keep growing in the background as more frees up, reserved into this process's own allocator before a competing process can grab it. Off by default.")
    parser.add_argument("--wait-for-vram-gb", type=float, default=None, help="Like --claim-vram-gb but BLOCKS at startup until this much is actually reserved before proceeding, instead of moving on with whatever's been claimed so far.")
    parser.add_argument("--wait-timeout", type=float, default=None, help="Give up waiting (raise TimeoutError) after this many seconds -- only applies with --wait-for-vram-gb. No timeout by default.")
    parser.add_argument("--keep-checkpoints", action="store_true", help="Added 2026-09-07: by default this script deletes each model's per-clip RDM/gram checkpoint once the final matrix is written (_clear_checkpoints()) -- fine for the summary matrix alone, but it destroys the only record of that model's own individual RDM, which a later self-RSA analysis (comparing one model's frozen vs. adapted RDM) needs. Pass this flag to keep them instead. Off by default so nothing about existing behavior changes unless explicitly requested.")
    args = parser.parse_args()
    main(args.category, args.condition, args.seed, args.fresh, args.claim_vram_gb, args.wait_for_vram_gb, args.wait_timeout, args.keep_checkpoints)
