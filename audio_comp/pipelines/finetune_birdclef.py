"""Matched LoRA-vs-frozen fine-tuning comparison for BirdCLEF (bird
sounds, 50-species classification), using
`xares_eval/birdclef/build_manifest.py`'s manifest + the pre-chopped 5s
clips already produced by `xares_eval/birdclef/make_audio_tar.py`'s
`_segment_recording()` (reused directly, not re-chopped here). Native
5-fold split (Stage 0's own fold assignment, species-balanced 4
recordings/species/fold).

Usage:
    python -m audio_comp.pipelines.finetune_birdclef --condition lora --model wav2vec2
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch

from audio_comp.pipelines.generic_lora_trainer import train_and_eval_allora, train_and_eval_frozen, train_and_eval_lora
from audio_comp.pipelines.allora_model_configs import ALLORA_SUPPORTED_MODELS
from audio_comp.pipelines.lora_model_configs import LORA_SUPPORTED_MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = REPO_ROOT / "data" / "birdclef_manifest.csv"
CLIPS_DIR = Path("/scratch/pdoshi/audio_comp/birdclef_clips")
NUM_FOLDS = 5
NATIVE_SAMPLE_RATE = 32000  # confirmed via xares_eval/birdclef/build_manifest.py's sf.write() -- preserves the HF dataset's own native rate, uniformly 32kHz for mteb/birdclef25-mini


def load_clips() -> tuple[list[dict], dict[str, int]]:
    with open(MANIFEST_CSV) as f:
        rows = list(csv.DictReader(f))
    species_list = sorted({r["species"] for r in rows})
    label_map = {s: i for i, s in enumerate(species_list)}

    clips = []
    for row in rows:
        n_clips = int(row["n_clips"])
        species_dir = CLIPS_DIR / row["species"]
        for i in range(n_clips):
            clip_path = species_dir / f"{row['row_index']}_{i:03d}.wav"
            if clip_path.exists():
                clips.append(dict(path=str(clip_path), label=label_map[row["species"]], fold=int(row["fold"])))
    return clips, label_map


def run_fold(condition: str, model_name: str, test_fold: int, device: str) -> float:
    clips, label_map = load_clips()
    train_pool = [c for c in clips if c["fold"] != test_fold]
    test_clips = [c for c in clips if c["fold"] == test_fold]

    rng = np.random.default_rng(0)
    val_idx = set(rng.choice(len(train_pool), size=max(1, int(0.15 * len(train_pool))), replace=False))
    train_clips = [c for i, c in enumerate(train_pool) if i not in val_idx]
    val_clips = [c for i, c in enumerate(train_pool) if i in val_idx]

    num_classes = len(label_map)
    if condition == "lora":
        return train_and_eval_lora(model_name, train_clips, val_clips, test_clips, num_classes, NATIVE_SAMPLE_RATE, device)
    if condition == "allora":
        return train_and_eval_allora(model_name, train_clips, val_clips, test_clips, num_classes, NATIVE_SAMPLE_RATE, device)
    return train_and_eval_frozen(model_name, train_clips, val_clips, test_clips, num_classes, device)


def main(condition: str, model_name: str) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_accs = []
    for test_fold in range(NUM_FOLDS):
        acc = run_fold(condition, model_name, test_fold, device)
        print(f"[{condition}:{model_name}] held-out fold={test_fold} test accuracy: {acc:.4f}")
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    print(f"[{condition}:{model_name}] mean accuracy across {NUM_FOLDS} folds: {mean_acc:.4f}")

    out_csv = REPO_ROOT / "results" / "finetune_birdclef.csv"
    write_header = not out_csv.exists()
    with open(out_csv, "a", newline="") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["model", "condition", "fold", "accuracy"])
        for fold, acc in enumerate(fold_accs):
            writer.writerow([model_name, condition, fold, acc])
        writer.writerow([model_name, condition, "mean", mean_acc])


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
