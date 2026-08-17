"""Matched LoRA-vs-frozen fine-tuning comparison for UrbanSound8K (city
noise, 10-class), using `xares_eval/urbansound8k/build_manifest.py`'s
manifest and the dataset's own native 10-fold CV (Salamon et al. 2014's
standard protocol).

**Subsampled to 4 held-out folds (1/4/7/10), not all 10** -- same
cost-bounding reasoning as MIMII's 4-of-16-fold choice: 10 folds x 14
LoRA models x several epochs each is not a bounded compute budget on top
of everything else already running. Read results as this 4-fold subset,
not full leave-one-fold-out CV.

Usage:
    python -m audio_comp.pipelines.finetune_urbansound8k --condition lora --model wav2vec2
"""
from __future__ import annotations

import csv
from pathlib import Path

import torch

from audio_comp.pipelines.generic_lora_trainer import train_and_eval_frozen, train_and_eval_lora
from audio_comp.pipelines.lora_model_configs import LORA_SUPPORTED_MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = REPO_ROOT / "data" / "urbansound8k_manifest.csv"
CLASS_LABEL_MAPS = {
    "air_conditioner": 0, "car_horn": 1, "children_playing": 2, "dog_bark": 3, "drilling": 4,
    "engine_idling": 5, "gun_shot": 6, "jackhammer": 7, "siren": 8, "street_music": 9,
}
NUM_CLASSES = len(CLASS_LABEL_MAPS)
NATIVE_SAMPLE_RATE = 44100  # UrbanSound8K clips are pre-decoded at their original sample rate by build_manifest.py; verify per-clip if this assumption ever breaks
HELD_OUT_FOLDS = [1, 4, 7, 10]
VAL_FRACTION_OF_TRAIN = 0.15


def load_clips() -> list[dict]:
    with open(MANIFEST_CSV) as f:
        rows = list(csv.DictReader(f))
    return [dict(path=r["file"], label=CLASS_LABEL_MAPS[r["soundevent"]], fold=int(r["fold"])) for r in rows]


def run_fold(condition: str, model_name: str, test_fold: int, device: str) -> float:
    clips = load_clips()
    train_pool = [c for c in clips if c["fold"] != test_fold]
    test_clips = [c for c in clips if c["fold"] == test_fold]

    import numpy as np

    rng = np.random.default_rng(0)
    val_idx = set(rng.choice(len(train_pool), size=int(VAL_FRACTION_OF_TRAIN * len(train_pool)), replace=False))
    train_clips = [c for i, c in enumerate(train_pool) if i not in val_idx]
    val_clips = [c for i, c in enumerate(train_pool) if i in val_idx]

    if condition == "lora":
        return train_and_eval_lora(model_name, train_clips, val_clips, test_clips, NUM_CLASSES, NATIVE_SAMPLE_RATE, device)
    return train_and_eval_frozen(model_name, train_clips, val_clips, test_clips, NUM_CLASSES, device)


def main(condition: str, model_name: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_accs = []
    for test_fold in HELD_OUT_FOLDS:
        acc = run_fold(condition, model_name, test_fold, device)
        print(f"[{condition}:{model_name}] held-out fold={test_fold} test accuracy: {acc:.4f}")
        fold_accs.append(acc)

    import numpy as np

    mean_acc = float(np.mean(fold_accs))
    print(f"[{condition}:{model_name}] mean accuracy across {len(HELD_OUT_FOLDS)} folds: {mean_acc:.4f}")

    out_csv = REPO_ROOT / "results" / "finetune_urbansound8k.csv"
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "condition", "fold", "accuracy"])
        for fold, acc in zip(HELD_OUT_FOLDS, fold_accs):
            writer.writerow([model_name, condition, fold, acc])
        writer.writerow([model_name, condition, "mean", mean_acc])


if __name__ == "__main__":
    import argparse

    import yaml

    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        active_models = sorted(yaml.safe_load(f)["active_models"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["frozen", "lora"], required=True)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()
    if args.condition == "lora" and args.model not in LORA_SUPPORTED_MODELS:
        raise ValueError(f"'{args.model}' not LoRA-compatible: {LORA_SUPPORTED_MODELS}")
    if args.condition == "frozen" and args.model not in active_models:
        raise ValueError(f"'{args.model}' not an active model: {active_models}")
    main(args.condition, args.model)
