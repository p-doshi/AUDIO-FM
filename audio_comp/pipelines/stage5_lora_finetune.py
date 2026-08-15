"""Stage 5 v1: matched LoRA fine-tuning on DeepShip, the cheapest real OOD
(underwater-acoustic) preview available given ShipsEar/DeepShip's full
datasets are both access-gated (see journal.md, 2026-08-10 DeepShip
entries). Per CLAUDE.md's Stage 5 methodology: this is the **matched**
fine-tuning condition (identical protocol across every model: same
dataset/split, epochs, optimizer/scheduler, trainable-layer policy,
seeds) -- the only condition used for the actual scientific comparison,
never blended with any official-fine-tuned-checkpoint condition.

**Scoped to 5 models this first pass, not the full roster --
architecture-driven, not arbitrary**: `wav2vec2`, `hubert`, `mert`,
`music2vec` share identical attention submodule names
(`q_proj`/`k_proj`/`v_proj`/`out_proj`, confirmed 2026-08-10 by directly
inspecting `named_modules()`, not assumed from architecture family
alone) and `ast` uses the same `q_proj`/`v_proj` names -- a genuinely
matched LoRA target across all 5, not five different approximations of
"the same policy." Excluded this pass: `panns_cnn14` (pure CNN, no
attention layer for a standard LoRA target to apply to at all -- would
need a fundamentally different adaptation technique, not a naming
difference); `clap`/`bird_mae`/`musicfm`/`audio_jepa`/`audiomae` (each
uses a different attention naming convention -- `query`/`value`, fused
`qkv`, `linear_q`/`linear_v`, or a flash_attn-shimmed custom module --
individually addable later but each needs its own verified target_modules
and its own smoke test, not assumed to work by analogy). Extending
coverage is real, bounded follow-up work, not deferred indefinitely.

LoRA config held identical across all 5 models: rank=8, alpha=16,
target_modules=["q_proj", "v_proj"], dropout=0.05. Trainable: LoRA
adapters + a fresh linear classification head (embed_dim -> 4 classes)
on top of the encoder's existing pooling convention (reuses
audio_comp.models adapters' own embed_batch()-equivalent pooling, so the
representation being adapted is the same one used everywhere else in
this project). Frozen: every other encoder weight.

Data: DeepShip's 63-clip GitHub subset, already segmented into 10s clips
(`xares_eval/deepship/make_audio_tar.py`'s output,
`$SCRATCH/audio_comp/deepship_clips/`) and fold-assigned
(`data/deepship_manifest.csv`, file-level grouping -- see that file's
docstring for why not vessel-level). 3-fold CV: train on 2 folds, LoRA
+ head only (base weights frozen), evaluate on the held-out fold; repeat
for all 3 fold rotations; report mean test accuracy per model.

Usage:
    python -m audio_comp.pipelines.stage5_lora_finetune --model wav2vec2
"""
from __future__ import annotations

import csv
import glob
import os
from collections import defaultdict
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from audio_comp.models import get_model_class
from audio_comp.models._util import mean_pool, resample

REPO_ROOT = Path(__file__).resolve().parents[2]
CLIPS_DIR = Path(os.environ.get("DEEPSHIP_CLIPS_DIR", "/scratch/pdoshi/audio_comp/deepship_clips"))
NUM_FOLDS = 3
NUM_CLASSES = 4
CLASS_LABEL_MAPS = {"cargo": 0, "passengership": 1, "tanker": 2, "tug": 3}

# Verified 2026-08-10 by direct inspection of named_modules() -- not
# assumed from architecture family. All 5 share the exact same target
# names, a genuinely matched LoRA policy, not five approximations.
LORA_TARGET_MODULES = ["q_proj", "v_proj"]
SUPPORTED_MODELS = ["wav2vec2", "hubert", "mert", "music2vec", "ast"]

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05
EPOCHS = 10
LEARNING_RATE = 1e-4
BATCH_SIZE = 8
SEED = 0


def load_clip_fold_map(manifest_csv: str) -> dict[str, int]:
    """record_id -> fold, from the DeepShip manifest."""
    with open(manifest_csv) as f:
        return {row["record_id"]: int(row["fold"]) for row in csv.DictReader(f)}


def build_clip_index() -> list[dict]:
    fold_map = load_clip_fold_map(str(REPO_ROOT / "data" / "deepship_manifest.csv"))
    rows = []
    for vessel_class in CLASS_LABEL_MAPS:
        for path in sorted(glob.glob(str(CLIPS_DIR / vessel_class / "*.wav"))):
            record_id = Path(path).stem.rsplit("_", 1)[0]
            rows.append(
                dict(
                    path=path,
                    vessel_class=vessel_class,
                    label=CLASS_LABEL_MAPS[vessel_class],
                    fold=fold_map[record_id],
                )
            )
    return rows


class LoraClassifier(nn.Module):
    def __init__(self, peft_model: nn.Module, embed_dim: int, num_classes: int, pooling_fn):
        super().__init__()
        self.peft_model = peft_model
        self.head = nn.Linear(embed_dim, num_classes)
        self.pooling_fn = pooling_fn

    def forward(self, **inputs) -> torch.Tensor:
        out = self.peft_model(**inputs)
        pooled = self.pooling_fn(out.last_hidden_state)
        return self.head(pooled)


def _pooler_output_forward(peft_model, **inputs):
    return peft_model(**inputs).pooler_output


def build_model_and_head(
    model_name: str, device: str, num_classes: int = NUM_CLASSES
) -> tuple[nn.Module, callable, int, int]:
    """Returns (trainable_module, forward_fn(module, inputs)->logits, sample_rate, embed_dim).

    forward_fn closes over trainable["head"] by reference (not a copy), so
    callers needing a different num_classes than this module's own
    DeepShip-specific default must pass it here rather than swapping
    trainable["head"] after the fact -- the closure would keep using the
    original head, silently ignoring the swap.
    """
    adapter = get_model_class(model_name)(device=device)
    adapter.load()

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=LORA_TARGET_MODULES,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    peft_model = get_peft_model(adapter._model, lora_config).to(device)
    embed_dim = peft_model.config.hidden_size
    trainable = nn.ModuleDict({"peft_model": peft_model, "head": nn.Linear(embed_dim, num_classes).to(device)})

    if model_name == "ast":

        def forward_fn(inputs):
            pooled = trainable["peft_model"](**inputs).pooler_output
            return trainable["head"](pooled)

    else:

        def forward_fn(inputs):
            hidden = trainable["peft_model"](**inputs).last_hidden_state
            pooled = mean_pool(hidden)
            return trainable["head"](pooled)

    return trainable, forward_fn, adapter.info.expected_sample_rate, embed_dim, adapter


def prepare_inputs(adapter, model_name: str, waveforms: list[np.ndarray], sample_rate: int, device: str) -> dict:
    resampled = [resample(w, sample_rate, adapter.info.expected_sample_rate) for w in waveforms]
    if model_name == "ast":
        inputs = adapter._extractor(resampled, sampling_rate=adapter.info.expected_sample_rate, return_tensors="pt")
    else:
        inputs = adapter._extractor(
            resampled, sampling_rate=adapter.info.expected_sample_rate, return_tensors="pt", padding=True
        )
    return {k: v.to(device) for k, v in inputs.items()}


def run_fold(model_name: str, test_fold: int, device: str) -> float:
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    clips = build_clip_index()
    train_clips = [c for c in clips if c["fold"] != test_fold]
    test_clips = [c for c in clips if c["fold"] == test_fold]

    trainable, forward_fn, sample_rate, embed_dim, adapter = build_model_and_head(model_name, device)
    optimizer = torch.optim.AdamW(
        [p for p in trainable.parameters() if p.requires_grad], lr=LEARNING_RATE
    )
    criterion = nn.CrossEntropyLoss()

    rng = np.random.default_rng(SEED)
    trainable.train()
    for epoch in range(EPOCHS):
        order = rng.permutation(len(train_clips))
        total_loss = 0.0
        for start in range(0, len(order), BATCH_SIZE):
            batch_idx = order[start : start + BATCH_SIZE]
            batch = [train_clips[i] for i in batch_idx]
            waveforms = [sf.read(c["path"], dtype="float32")[0] for c in batch]
            labels = torch.tensor([c["label"] for c in batch], device=device)

            inputs = prepare_inputs(adapter, model_name, waveforms, sample_rate, device)
            logits = forward_fn(inputs)
            loss = criterion(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
        print(f"  [{model_name} fold={test_fold}] epoch {epoch + 1}/{EPOCHS} loss={total_loss / len(train_clips):.4f}", flush=True)

    trainable.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for start in range(0, len(test_clips), BATCH_SIZE):
            batch = test_clips[start : start + BATCH_SIZE]
            waveforms = [sf.read(c["path"], dtype="float32")[0] for c in batch]
            labels = torch.tensor([c["label"] for c in batch], device=device)
            inputs = prepare_inputs(adapter, model_name, waveforms, sample_rate, device)
            logits = forward_fn(inputs)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += len(batch)

    return correct / total


def main(model_name: str) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model '{model_name}' not in Stage 5 v1's supported set: {SUPPORTED_MODELS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_accs = []
    for test_fold in range(NUM_FOLDS):
        acc = run_fold(model_name, test_fold, device)
        print(f"[{model_name}] fold {test_fold} test accuracy: {acc:.4f}")
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    print(f"[{model_name}] mean accuracy across {NUM_FOLDS} folds: {mean_acc:.4f}")

    out_csv = REPO_ROOT / "results" / "stage5_lora_deepship.csv"
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "fold", "accuracy"])
        for fold, acc in enumerate(fold_accs):
            writer.writerow([model_name, fold, acc])
        writer.writerow([model_name, "mean", mean_acc])


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    args = parser.parse_args()
    main(args.model)
