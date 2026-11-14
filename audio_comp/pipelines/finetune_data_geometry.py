"""Frozen (pretrained) representational geometry computed ON EACH
FINE-TUNING DATASET'S OWN CLIPS, instead of on the shared 16k probe set.

Why this exists
---------------
Finding 5 correlates each model's *probe-set* uniformity against that
model's LoRA/ALLoRA/top-K adaptation gain on a downstream task. The
uniformity side is therefore measured on a different data distribution
than the gain side. That is deliberate (Stage 6's whole premise is a
cheap label-free probe that predicts expensive labelled outcomes), but
it was never checked whether uniformity is actually distribution-stable
-- i.e. whether a model's uniformity computed on the FMA-genre clips,
the UrbanSound8K clips, etc. ranks the 19 models the same way the
probe-set uniformity does. This script produces the task-matched
numbers so that check can be run.

What it computes, per (dataset, model), on FROZEN embeddings only
(no training, no seeds -- frozen extraction is deterministic):
  - uniformity   (Wang & Isola 2020, the Finding 5 metric)
  - alignment    (same-label positive-pair proxy, alignment_score())
  - twonn        (TwoNN intrinsic dimension)
and, across models per dataset (--finalize):
  - RSA matrix   (Spearman between correlation-RDMs, matched clip order)
  - CKA matrix   (linear CKA, matched clip order)

Datasets: the 5 public fine-tuning categories whose manifests live
locally -- fma_genre, urbansound8k, birdclef, librispeech, mimii. The
confidential vessel domain is NOT here: its geometry-on-fine-tuning-data
equivalent is the restricted-domain variant of this pipeline (--condition frozen would need
adding there) and is run separately on a restricted server.

Clip set per dataset = the FULL pooled manifest (train+val+test), the
same clips the matched fine-tuning actually sees, capped at --max-clips
(default 4000) via a single fixed-seed subsample shared across every
model so RSA/CKA rows line up. Undecodable clips (a few FMA mp3s) are
dropped per-model and realigned by clip path at --finalize time, same
as the probe-set npz pipeline.

Env vars the underlying loaders need on the secondary cluster (unchanged from the
finetune_*.py scripts):
  BIRDCLEF_CLIPS_DIR              (birdclef -- the secondary cluster has no /scratch)
  LIBRISPEECH_SPEAKER_MANIFEST_CSV / LIBRISPEECH_SPEAKER_AUDIO_DIR
  MIMII_RAW_DIR

Usage
-----
  # one model on one dataset
  python -m audio_comp.pipelines.finetune_data_geometry --dataset fma_genre --model wav2vec2

  # all active models on one dataset, then build the matrices
  python -m audio_comp.pipelines.finetune_data_geometry --dataset fma_genre --all-models --finalize

  # everything, one process, one pass  (the "run 19 models once" path)
  python -m audio_comp.pipelines.finetune_data_geometry --all-datasets --all-models --finalize
"""
from __future__ import annotations

import argparse
import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml

from audio_comp.models import get_model_class
from audio_comp.geometry.alignment_uniformity import alignment_score, uniformity_score
from audio_comp.geometry.intrinsic_dim import twonn_intrinsic_dimension
from audio_comp.geometry.rdm import compute_rdm
from audio_comp.geometry.rsa import rsa_score
from audio_comp.pipelines._vram import GreedyVRAMClaimer
from audio_comp.pipelines.generic_lora_trainer import read_batch_skip_bad, write_results_idempotent
from audio_comp.pipelines.rsa_cka_finetuned import CATEGORY_LOADERS

REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = REPO_ROOT / "results"
EMB_DIR = Path(
    # keep the cached embeddings off the repo; same $SCRATCH convention as
    # the probe-set embeddings, overridable for the secondary cluster (no /scratch there).
    __import__("os").environ.get(
        "FINETUNE_GEOM_EMB_DIR", str(REPO_ROOT / "results" / "finetune_data_geometry_emb")
    )
)
BATCH_SIZE = 16
SUBSAMPLE_SEED = 0  # the ONE seed in this script: which clips enter the capped set. Shared across all models.
# 2026-09-07: wav2vec2_conformer's relative-position attention OOMs at BATCH_SIZE=16 on
# this pipeline too (the same reason it's missing from the mimii RSA/CKA matrix, 18/19
# models) -- same fix already proven for it elsewhere in this project (stage5_lora/
# allora_finetune_mimii.py's MODEL_BATCH_SIZE_OVERRIDES, the restricted-domain variant's).
MODEL_BATCH_SIZE_OVERRIDES = {"wav2vec2_conformer": 4}
_PINNED = False  # set True once a GreedyVRAMClaimer has reserved a pool into this process's allocator
                 # -- then skip torch.cuda.empty_cache() so the reservation isn't handed back to the OS.

DATASETS = ["fma_genre", "urbansound8k", "birdclef", "librispeech", "mimii"]


def active_models() -> list[str]:
    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        return sorted(yaml.safe_load(f)["active_models"])


def _capped_clip_list(all_clips: list[dict], max_clips: int) -> list[dict]:
    """Deterministic: sort by path, then one fixed-seed subsample. Identical
    for every model so the --finalize RSA/CKA rows are matched."""
    clips_sorted = sorted(all_clips, key=lambda c: c["path"])
    if len(clips_sorted) <= max_clips:
        return clips_sorted
    rng = np.random.default_rng(SUBSAMPLE_SEED)
    idx = np.sort(rng.choice(len(clips_sorted), size=max_clips, replace=False))
    return [clips_sorted[i] for i in idx]


def _emb_path(dataset: str, model_name: str) -> Path:
    return EMB_DIR / f"{dataset}__{model_name}.npz"


# ---------------------------------------------------------------------------
# Per-(dataset, model) frozen extraction + metrics
# ---------------------------------------------------------------------------

def extract_one(dataset: str, model_name: str, max_clips: int, device: str, overwrite: bool) -> None:
    out_npz = _emb_path(dataset, model_name)
    if out_npz.exists() and not overwrite:
        print(f"[{dataset}:{model_name}] npz exists, skipping extraction (use --overwrite to redo)", flush=True)
    else:
        all_clips, _, _, num_classes, native_sr, max_dur = CATEGORY_LOADERS[dataset]()
        clips = _capped_clip_list(all_clips, max_clips)
        print(f"[{dataset}:{model_name}] {len(clips)} clips (of {len(all_clips)} pooled), "
              f"native_sr={native_sr} max_dur={max_dur} num_classes={num_classes}", flush=True)

        adapter = get_model_class(model_name)(device=device)
        adapter.load()
        batch_size = MODEL_BATCH_SIZE_OVERRIDES.get(model_name, BATCH_SIZE)
        emb_chunks, kept_ids, kept_labels = [], [], []
        for start in range(0, len(clips), batch_size):
            batch = clips[start : start + batch_size]
            waveforms, kept = read_batch_skip_bad(batch, native_sr, max_dur)
            if not waveforms:
                continue
            emb = adapter.embed_batch(waveforms, native_sr)
            emb_chunks.append(np.asarray(emb, dtype=np.float32))
            kept_ids.extend(c["path"] for c in kept)
            kept_labels.extend(int(c["label"]) for c in kept)
            if device == "cuda" and not _PINNED:
                torch.cuda.empty_cache()
            if (start // BATCH_SIZE) % 25 == 0:
                print(f"  {start + len(batch)}/{len(clips)}", flush=True)
        del adapter
        if device == "cuda" and not _PINNED:
            torch.cuda.empty_cache()

        embeddings = np.concatenate(emb_chunks, axis=0)
        clip_ids = np.array(kept_ids)
        labels = np.array(kept_labels)
        EMB_DIR.mkdir(parents=True, exist_ok=True)
        np.savez(out_npz, clip_ids=clip_ids, embeddings=embeddings, labels=labels)
        print(f"[{dataset}:{model_name}] saved {embeddings.shape} -> {out_npz}", flush=True)

    _write_per_model_metrics(dataset, model_name, out_npz)


def _write_per_model_metrics(dataset: str, model_name: str, npz_path: Path) -> None:
    data = np.load(npz_path, allow_pickle=True)
    emb, labels = data["embeddings"], data["labels"]
    uni = uniformity_score(emb, sample_size=len(emb))  # use every clip; n<=max_clips is cheap
    ali = alignment_score(emb, labels)
    try:
        tnn = twonn_intrinsic_dimension(emb)
    except Exception as e:  # skdim occasionally chokes on degenerate neighbourhoods
        print(f"  WARNING twonn failed for {dataset}:{model_name}: {repr(e)[:120]}", flush=True)
        tnn = float("nan")
    print(f"[{dataset}:{model_name}] uniformity={uni:.4f} alignment={ali:.4f} twonn={tnn:.3f} n={len(emb)}", flush=True)
    write_results_idempotent(
        RESULTS_DIR / "finetune_data_geometry.csv",
        ["dataset", "model", "n_clips", "uniformity", "alignment", "twonn"],
        ["dataset", "model"],
        [[dataset, model_name, len(emb), f"{uni:.6f}", f"{ali:.6f}", f"{tnn:.6f}"]],
    )


# ---------------------------------------------------------------------------
# --finalize: matched-clip RSA + CKA matrices, plus matched-set metrics
# ---------------------------------------------------------------------------

def finalize(dataset: str) -> None:
    npzs = sorted(EMB_DIR.glob(f"{dataset}__*.npz"))
    if len(npzs) < 2:
        print(f"[{dataset}] only {len(npzs)} model npz(s) present, need >=2 to build matrices -- skipping finalize", flush=True)
        return
    loaded = {}
    for p in npzs:
        model = p.name[len(dataset) + 2 : -4]
        d = np.load(p, allow_pickle=True)
        loaded[model] = (list(map(str, d["clip_ids"])), d["embeddings"], d["labels"])

    common = set.intersection(*[set(ids) for ids, _, _ in loaded.values()])
    common_ordered = sorted(common)
    print(f"[{dataset}] {len(loaded)} models, {len(common_ordered)} clips common to all", flush=True)

    models = sorted(loaded)
    aligned_emb, matched_metrics = {}, []
    for m in models:
        ids, emb, labels = loaded[m]
        pos = {cid: i for i, cid in enumerate(ids)}
        rows = [pos[c] for c in common_ordered]
        e = emb[rows]
        aligned_emb[m] = e
        lab = labels[rows]
        uni = uniformity_score(e, sample_size=len(e))
        ali = alignment_score(e, lab)
        try:
            tnn = twonn_intrinsic_dimension(e)
        except Exception:
            tnn = float("nan")
        matched_metrics.append([dataset, m, len(e), f"{uni:.6f}", f"{ali:.6f}", f"{tnn:.6f}"])

    write_results_idempotent(
        RESULTS_DIR / "finetune_data_geometry_matched.csv",
        ["dataset", "model", "n_clips", "uniformity", "alignment", "twonn"],
        ["dataset", "model"],
        matched_metrics,
    )

    rdms = {m: compute_rdm(aligned_emb[m]) for m in models}
    n = len(models)
    rsa_mat = np.eye(n)
    cka_mat = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            rsa_mat[i, j] = rsa_mat[j, i] = rsa_score(rdms[models[i]], rdms[models[j]])
            cka_mat[i, j] = cka_mat[j, i] = _linear_cka(aligned_emb[models[i]], aligned_emb[models[j]])

    _write_matrix(RESULTS_DIR / f"finetune_data_geometry_rsa_{dataset}.csv", models, rsa_mat)
    _write_matrix(RESULTS_DIR / f"finetune_data_geometry_cka_{dataset}.csv", models, cka_mat)
    _heatmap(RESULTS_DIR / f"finetune_data_geometry_rsa_{dataset}.png", models, rsa_mat, f"RSA (frozen) -- {dataset}")
    print(f"[{dataset}] wrote RSA/CKA matrices + heatmap for {n} models", flush=True)


def _linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    m = x.shape[0]
    c = np.eye(m) - np.ones((m, m)) / m
    gx = c @ (x @ x.T) @ c
    gy = c @ (y @ y.T) @ c
    num = np.sum(gx * gy)
    den = np.sqrt(np.sum(gx * gx) * np.sum(gy * gy))
    return float(num / den) if den else float("nan")


def _write_matrix(path: Path, models: list[str], mat: np.ndarray) -> None:
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model"] + models)
        for m, row in zip(models, mat):
            w.writerow([m] + [f"{v:.6f}" for v in row])


def _heatmap(path: Path, models: list[str], mat: np.ndarray, title: str) -> None:
    fig, ax = plt.subplots(figsize=(0.55 * len(models) + 2, 0.55 * len(models) + 2))
    im = ax.imshow(mat, vmin=-0.1, vmax=1.0, cmap="viridis")
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels(models, rotation=90, fontsize=7)
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels(models, fontsize=7)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", choices=DATASETS)
    ap.add_argument("--all-datasets", action="store_true")
    ap.add_argument("--model")
    ap.add_argument("--all-models", action="store_true")
    ap.add_argument("--max-clips", type=int, default=4000)
    ap.add_argument("--overwrite", action="store_true", help="re-extract even if the npz already exists")
    ap.add_argument("--finalize", action="store_true", help="after extraction, build the matched RSA/CKA matrices per dataset")
    ap.add_argument("--finalize-only", action="store_true", help="skip extraction, just (re)build matrices from existing npz")
    ap.add_argument("--wait-for-vram-gb", type=float, default=None,
                    help="Block at startup until this many GB are free on the GPU (index 0 as this process sees it -- "
                         "pick the physical GPU with CUDA_VISIBLE_DEVICES), claim it progressively, then release it "
                         "into this process's allocator before loading any model -- avoids racing a co-tenant and OOMing.")
    ap.add_argument("--wait-timeout", type=float, default=None, help="Give up waiting (TimeoutError) after this many seconds; default: wait forever.")
    ap.add_argument("--claim-vram-gb", type=float, default=None,
                    help="Like --wait-for-vram-gb but non-blocking: grab whatever is free now (up to this cap), keep "
                         "growing in the background, don't wait for the full amount before starting.")
    args = ap.parse_args()

    # Pinned to physical GPU 0 explicitly (2026-09-03, per explicit user
    # request, same change/reasoning as the restricted-domain scripts).
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    global _PINNED
    target_gb = args.wait_for_vram_gb or args.claim_vram_gb
    if device == "cuda" and target_gb:
        claimer = GreedyVRAMClaimer(device=0, max_gb=target_gb).start()
        if args.wait_for_vram_gb:
            claimer.wait_until_target(timeout=args.wait_timeout)
        claimer.release()  # hand the reserved pool to our own allocator; the loop below reuses it
        _PINNED = True
    datasets = DATASETS if args.all_datasets else ([args.dataset] if args.dataset else [])
    if not datasets:
        ap.error("pass --dataset <name> or --all-datasets")
    models = active_models() if args.all_models else ([args.model] if args.model else [])
    if not models and not args.finalize_only:
        ap.error("pass --model <name>, --all-models, or --finalize-only")

    # OOM-retry-with-wait (2026-09-03, per explicit user request, same
    # pattern as the restricted-domain scripts):
    # a CUDA OOM no longer just skips to the next model -- it clears cache,
    # polls live free memory every 30s, and retries the SAME model,
    # indefinitely, no retry cap. Non-OOM exceptions still just log and
    # move on (retrying those would reproduce the same failure forever).
    MIN_FREE_GB_TO_RETRY = 20.0
    POLL_INTERVAL_S = 30.0

    def extract_with_oom_retry(ds: str, mdl: str) -> None:
        attempt = 0
        while True:
            try:
                extract_one(ds, mdl, args.max_clips, device, args.overwrite)
                return
            except torch.cuda.OutOfMemoryError as e:
                attempt += 1
                torch.cuda.empty_cache()
                print(f"[{ds}:{mdl}] [oom-retry] attempt {attempt} hit CUDA OOM ({e}) -- "
                      f"waiting for >={MIN_FREE_GB_TO_RETRY:.0f}GB free on {device} before retrying "
                      f"(checking every {POLL_INTERVAL_S:.0f}s, no retry limit)", flush=True)
                last_print = 0.0
                while True:
                    free_bytes, _total = torch.cuda.mem_get_info(0)
                    free_gb = free_bytes / (1024 ** 3)
                    if free_gb >= MIN_FREE_GB_TO_RETRY:
                        print(f"[{ds}:{mdl}] [oom-retry] {free_gb:.1f}GB free now -- retrying", flush=True)
                        break
                    now = time.monotonic()
                    if now - last_print >= 300:
                        print(f"[{ds}:{mdl}] [oom-retry] still waiting, {free_gb:.1f}GB free "
                              f"(need {MIN_FREE_GB_TO_RETRY:.0f}GB)", flush=True)
                        last_print = now
                    time.sleep(POLL_INTERVAL_S)

    for ds in datasets:
        if not args.finalize_only:
            for mdl in models:
                try:
                    extract_with_oom_retry(ds, mdl)
                except Exception as e:
                    print(f"[{ds}:{mdl}] FAILED (non-OOM): {repr(e)[:300]}", flush=True)
        if args.finalize or args.finalize_only:
            finalize(ds)


if __name__ == "__main__":
    main()
