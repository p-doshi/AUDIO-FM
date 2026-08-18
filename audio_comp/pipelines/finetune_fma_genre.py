"""Matched LoRA-vs-frozen fine-tuning comparison for FMA-genre (music,
8-class), using `xares_eval/fma_genre/build_manifest.py`'s manifest and
FMA-small's own native 80/10/10 train/validation/test split -- one run,
not a k-fold loop, since the dataset's own split isn't fold-based.

Usage:
    python -m audio_comp.pipelines.finetune_fma_genre --condition lora --model wav2vec2
    python -m audio_comp.pipelines.finetune_fma_genre --condition frozen --model wav2vec2
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from audio_comp.pipelines.generic_lora_trainer import train_and_eval_allora, train_and_eval_frozen, train_and_eval_lora
from audio_comp.pipelines.allora_model_configs import ALLORA_SUPPORTED_MODELS
from audio_comp.pipelines.lora_model_configs import LORA_SUPPORTED_MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = REPO_ROOT / "data" / "fma_genre_manifest.csv"
GENRE_CLASSES = sorted(
    {"Hip-Hop", "Pop", "Folk", "Experimental", "Rock", "International", "Electronic", "Instrumental"}
)
LABEL_MAP = {g: i for i, g in enumerate(GENRE_CLASSES)}
NUM_CLASSES = len(LABEL_MAP)
NATIVE_SAMPLE_RATE = 44100  # target rate every clip gets resampled to (per-file true rate handled by read_mono(), not assumed) -- FMA-small's own mp3s are uniformly 44.1kHz so this also happens to be the native rate here
MAX_CLIP_DURATION_S = 10.0  # FMA-small's 30s clips OOM'd multiple large models on a full 80GB GPU during fine-tuning; matches X-ARES's own upstream fma_genre_config crop_length=10


def load_clips() -> tuple[list[dict], list[dict], list[dict]]:
    with open(MANIFEST_CSV) as f:
        rows = list(csv.DictReader(f))
    by_split: dict[str, list[dict]] = {"training": [], "validation": [], "test": []}
    for row in rows:
        by_split[row["split"]].append(dict(path=row["file"], label=LABEL_MAP[row["genre"]]))
    return by_split["training"], by_split["validation"], by_split["test"]


def main(condition: str, model_name: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_clips, val_clips, test_clips = load_clips()
    print(f"train={len(train_clips)} val={len(val_clips)} test={len(test_clips)}")

    if condition == "lora":
        acc = train_and_eval_lora(model_name, train_clips, val_clips, test_clips, NUM_CLASSES, NATIVE_SAMPLE_RATE, device, max_duration_s=MAX_CLIP_DURATION_S)
    elif condition == "allora":
        acc = train_and_eval_allora(model_name, train_clips, val_clips, test_clips, NUM_CLASSES, NATIVE_SAMPLE_RATE, device, max_duration_s=MAX_CLIP_DURATION_S)
    else:
        acc = train_and_eval_frozen(model_name, train_clips, val_clips, test_clips, NUM_CLASSES, device, native_sample_rate=NATIVE_SAMPLE_RATE, max_duration_s=MAX_CLIP_DURATION_S)

    print(f"[{condition}:{model_name}] test accuracy: {acc:.4f}")

    out_csv = REPO_ROOT / "results" / "finetune_fma_genre.csv"
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
