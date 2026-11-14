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
import os
from pathlib import Path

import numpy as np
import torch

from audio_comp.pipelines.generic_lora_trainer import (
    train_and_eval_allora,
    train_and_eval_frozen,
    train_and_eval_lora,
    train_and_eval_topk,
    write_results_idempotent,
)
from audio_comp.pipelines.allora_model_configs import ALLORA_SUPPORTED_MODELS
from audio_comp.pipelines.lora_model_configs import LORA_SUPPORTED_MODELS
from audio_comp.pipelines.topk_unfreeze_configs import TOPK_SUPPORTED_MODELS

REPO_ROOT = Path(__file__).resolve().parents[2]
MANIFEST_CSV = REPO_ROOT / "data" / "birdclef_manifest.csv"
# Was hardcoded to data_cache/birdclef_clips -- broke
# immediately when this script was run on the secondary cluster (2026-08-27), which has no
# /scratch at all. Parametrized via env var, matching this project's
# existing convention for every other cross-cluster data path
# (MIMII_RAW_DIR, US8K_AUDIO_DIR, etc.) -- default preserves the exact
# original primary-cluster value, so nothing changes there.
CLIPS_DIR = Path(os.environ.get("BIRDCLEF_CLIPS_DIR", "data_cache/birdclef_clips"))
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


def run_fold(condition: str, model_name: str, test_fold: int, device: str, seed: int = 0) -> float:
    clips, label_map = load_clips()
    train_pool = [c for c in clips if c["fold"] != test_fold]
    test_clips = [c for c in clips if c["fold"] == test_fold]

    # `seed` (added 2026-08-20 for multi-seed variance runs) now also
    # drives the train/val split, not just model init.
    rng = np.random.default_rng(seed)
    val_idx = set(rng.choice(len(train_pool), size=max(1, int(0.15 * len(train_pool))), replace=False))
    train_clips = [c for i, c in enumerate(train_pool) if i not in val_idx]
    val_clips = [c for i, c in enumerate(train_pool) if i in val_idx]

    num_classes = len(label_map)
    predictions_out_csv = REPO_ROOT / "results" / "per_clip_predictions" / f"birdclef_{model_name}_{condition}_fold{test_fold}_seed{seed}.csv"
    if condition == "lora":
        return train_and_eval_lora(model_name, train_clips, val_clips, test_clips, num_classes, NATIVE_SAMPLE_RATE, device, predictions_out_csv=predictions_out_csv, seed=seed)
    if condition == "allora":
        return train_and_eval_allora(model_name, train_clips, val_clips, test_clips, num_classes, NATIVE_SAMPLE_RATE, device, predictions_out_csv=predictions_out_csv, seed=seed)
    if condition == "topk":
        return train_and_eval_topk(model_name, train_clips, val_clips, test_clips, num_classes, device, predictions_out_csv=predictions_out_csv, seed=seed)
    return train_and_eval_frozen(model_name, train_clips, val_clips, test_clips, num_classes, device, native_sample_rate=NATIVE_SAMPLE_RATE, predictions_out_csv=predictions_out_csv, seed=seed)


def main(condition: str, model_name: str, seed: int = 0) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_accs = []
    for test_fold in range(NUM_FOLDS):
        acc = run_fold(condition, model_name, test_fold, device, seed)
        print(f"[{condition}:{model_name}:seed{seed}] held-out fold={test_fold} test accuracy: {acc:.4f}")
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    print(f"[{condition}:{model_name}:seed{seed}] mean accuracy across {NUM_FOLDS} folds: {mean_acc:.4f}")

    out_csv = REPO_ROOT / "results" / "finetune_birdclef.csv"
    new_rows = [[model_name, condition, seed, fold, acc] for fold, acc in enumerate(fold_accs)]
    new_rows.append([model_name, condition, seed, "mean", mean_acc])
    write_results_idempotent(
        out_csv, ["model", "condition", "seed", "fold", "accuracy"], ["model", "condition", "seed", "fold"], new_rows
    )


if __name__ == "__main__":
    import argparse

    import yaml

    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        active_models = sorted(yaml.safe_load(f)["active_models"])

    parser = argparse.ArgumentParser()
    parser.add_argument("--condition", choices=["frozen", "lora", "allora", "topk"], required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=0, help="Random seed for train/val split + model init/training-order shuffling -- added 2026-08-20 for multi-seed variance runs. Seed 0 matches every prior result already on record.")
    args = parser.parse_args()
    if args.condition == "lora" and args.model not in LORA_SUPPORTED_MODELS:
        raise ValueError(f"'{args.model}' not LoRA-compatible: {LORA_SUPPORTED_MODELS}")
    if args.condition == "allora" and args.model not in ALLORA_SUPPORTED_MODELS:
        raise ValueError(f"'{args.model}' not ALLoRA-compatible: {ALLORA_SUPPORTED_MODELS}")
    if args.condition == "topk" and args.model not in TOPK_SUPPORTED_MODELS:
        raise ValueError(f"'{args.model}' not top-K-unfreeze-compatible: {TOPK_SUPPORTED_MODELS}")
    if args.condition == "frozen" and args.model not in active_models:
        raise ValueError(f"'{args.model}' not an active model: {active_models}")
    main(args.condition, args.model, args.seed)
