"""Matched LoRA-vs-frozen fine-tuning comparison for LibriSpeech
speaker-ID (speech, ~251-way closed-set classification), using
`xares_eval/librispeech_speaker/build_manifest.py`'s manifest and its own
per-speaker-stratified 70/15/15 train/val/test split (utterance-level
disjoint, full speaker overlap across splits by design -- see that
script's docstring for why this is the correct split for a closed-set
classification task, unlike DeepShip/vessel's held-out-entity splits).

Closes the one probe-set category (speech) that had no matched
LoRA/ALLoRA-vs-frozen comparison, and doubles as a direct test of a
pre-registered prediction (journal.md, 2026-08-18): BirdCLEF (50-way) was
the one real-signal task where Finding 5's uniformity-vs-adaptation-gain
effect went null while the coarser FMA-genre (8-way) and UrbanSound8K
(10-way) both showed it. LibriSpeech speaker-ID at ~251-way is an even
finer-grained task than BirdCLEF -- if task granularity (not something
bioacoustic-specific) drives the null, this should replicate it.

Usage:
    python -m audio_comp.pipelines.finetune_librispeech --condition lora --model wav2vec2
    python -m audio_comp.pipelines.finetune_librispeech --condition frozen --model wav2vec2
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from audio_comp.pipelines.generic_lora_trainer import train_and_eval_allora, train_and_eval_frozen, train_and_eval_lora
from audio_comp.pipelines.allora_model_configs import ALLORA_SUPPORTED_MODELS
from audio_comp.pipelines.lora_model_configs import LORA_SUPPORTED_MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = REPO_ROOT / "data" / "librispeech_speaker_manifest.csv"
NATIVE_SAMPLE_RATE = 16000  # openslr/librispeech_asr is uniformly 16kHz (confirmed via the dataset's own schema, same check used before building the manifest)
MAX_CLIP_DURATION_S = 10.0  # LibriSpeech utterances are typically a few seconds but occasionally longer; cap for consistency with every other fine-tuning script's OOM-safety convention


def load_clips() -> tuple[list[dict], list[dict], list[dict], int]:
    with open(MANIFEST_CSV) as f:
        rows = list(csv.DictReader(f))
    num_classes = len({row["label"] for row in rows})
    by_split: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    for row in rows:
        by_split[row["split"]].append(dict(path=row["file"], label=int(row["label"])))
    return by_split["train"], by_split["val"], by_split["test"], num_classes


def main(condition: str, model_name: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_clips, val_clips, test_clips, num_classes = load_clips()
    print(f"train={len(train_clips)} val={len(val_clips)} test={len(test_clips)} num_classes={num_classes}")

    if condition == "lora":
        acc = train_and_eval_lora(model_name, train_clips, val_clips, test_clips, num_classes, NATIVE_SAMPLE_RATE, device, max_duration_s=MAX_CLIP_DURATION_S)
    elif condition == "allora":
        acc = train_and_eval_allora(model_name, train_clips, val_clips, test_clips, num_classes, NATIVE_SAMPLE_RATE, device, max_duration_s=MAX_CLIP_DURATION_S)
    else:
        acc = train_and_eval_frozen(model_name, train_clips, val_clips, test_clips, num_classes, device, native_sample_rate=NATIVE_SAMPLE_RATE, max_duration_s=MAX_CLIP_DURATION_S)

    print(f"[{condition}:{model_name}] test accuracy: {acc:.4f}")

    out_csv = REPO_ROOT / "results" / "finetune_librispeech_speaker.csv"
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "condition", "accuracy"])
        writer.writerow([model_name, condition, acc])


if __name__ == "__main__":
    import argparse

    import yaml

    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        active_models = sorted(yaml.safe_load(f)["active_models"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["frozen", "lora", "allora"], required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.condition == "lora" and args.model not in LORA_SUPPORTED_MODELS:
        raise ValueError(f"'{args.model}' not LoRA-compatible: {LORA_SUPPORTED_MODELS}")
    if args.condition == "allora" and args.model not in ALLORA_SUPPORTED_MODELS:
        raise ValueError(f"'{args.model}' not ALLoRA-compatible: {ALLORA_SUPPORTED_MODELS}")
    if args.condition == "frozen" and args.model not in active_models:
        raise ValueError(f"'{args.model}' not an active model: {active_models}")
    main(args.condition, args.model)
