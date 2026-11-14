"""Top-K-layer-unfreezing counterpart to stage5_lora_finetune_mimii.py --
the 4th MIMII fine-tuning condition, for the 4 models that can't go
through LoRA at all (panns_cnn14/audio_jepa/audiomae/musicfm -- see
topk_unfreeze_configs.py's module docstring for why). Built 2026-09-03:
this condition was never implemented for MIMII at all (only
frozen/lora/allora existed) -- a real gap, not a silent skip, closed
here using the same train_and_eval_topk() the other 4 fine-tuning
domains (fma_genre/urbansound8k/birdclef/librispeech) already use,
rather than hand-rolling a separate two-param-group training loop.

Same leave-one-machine-type-out protocol as the other 3 MIMII
conditions (4 folds, build_clip_index()'s 2400-clip balanced pool as of
2026-09-03) -- differs from them only in adding a small validation split
carved out of the training pool, since train_and_eval_topk() (unlike
MIMII's simpler fixed-epoch lora/allora loops) does early-stopping model
selection against a validation set, matching how every other domain's
topk condition already works.

Usage:
    python -m audio_comp.pipelines.stage5_topk_finetune_mimii --model panns_cnn14
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from audio_comp.pipelines.generic_lora_trainer import train_and_eval_topk
from audio_comp.pipelines.topk_unfreeze_configs import TOPK_SUPPORTED_MODELS as SUPPORTED_MODELS
from audio_comp.pipelines.stage5_lora_finetune_mimii import (
    SEED,
    MIMII_NATIVE_SAMPLE_RATE,
    NUM_CLASSES,
    MACHINE_TYPES,
    NUM_FOLDS,
    build_clip_index,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
VAL_FRACTION_OF_TRAIN = 0.15


def run_fold(model_name: str, test_machine: str, device: str, seed: int = SEED) -> float:
    clips = build_clip_index(seed)
    train_pool = [c for c in clips if c["machine"] != test_machine]
    test_clips = [c for c in clips if c["machine"] == test_machine]

    # Small val split carved out of the training pool -- train_and_eval_topk()
    # does early-stopping model selection against it, unlike MIMII's other
    # 3 conditions' simpler fixed-epoch loops. Same seeded-rng convention
    # used throughout this project's multi-seed infrastructure.
    rng = np.random.default_rng(seed)
    val_idx = set(rng.choice(len(train_pool), size=max(1, int(VAL_FRACTION_OF_TRAIN * len(train_pool))), replace=False))
    train_clips = [c for i, c in enumerate(train_pool) if i not in val_idx]
    val_clips = [c for i, c in enumerate(train_pool) if i in val_idx]

    predictions_out_csv = REPO_ROOT / "results" / "per_clip_predictions" / f"mimii_{model_name}_topk_machine{test_machine}_seed{seed}.csv"
    return train_and_eval_topk(
        model_name, train_clips, val_clips, test_clips, NUM_CLASSES, device,
        predictions_out_csv=predictions_out_csv, seed=seed,
    )


def main(model_name: str, seed: int = SEED) -> None:
    if model_name not in SUPPORTED_MODELS:
        raise ValueError(f"model '{model_name}' not in top-K-unfreeze-verified set: {SUPPORTED_MODELS}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    fold_accs = []
    for test_machine in MACHINE_TYPES:
        acc = run_fold(model_name, test_machine, device, seed)
        print(f"[topk:{model_name}:seed{seed}] held-out machine={test_machine} test accuracy: {acc:.4f}")
        fold_accs.append(acc)

    mean_acc = float(np.mean(fold_accs))
    print(f"[topk:{model_name}:seed{seed}] mean accuracy across {NUM_FOLDS} leave-one-machine-out folds: {mean_acc:.4f}")

    from audio_comp.pipelines.generic_lora_trainer import write_results_idempotent
    out_csv = REPO_ROOT / "results" / "stage5_topk_mimii.csv"
    new_rows = [[model_name, seed, machine, acc] for machine, acc in zip(MACHINE_TYPES, fold_accs)]
    new_rows.append([model_name, seed, "mean", mean_acc])
    write_results_idempotent(
        out_csv, ["model", "seed", "held_out_machine", "accuracy"], ["model", "seed", "held_out_machine"], new_rows
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, choices=SUPPORTED_MODELS)
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for data subsampling + val split + model init -- seed 0 matches this project's convention elsewhere.")
    args = parser.parse_args()
    main(args.model, args.seed)
