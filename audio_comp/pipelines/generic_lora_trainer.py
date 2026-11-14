"""Shared LoRA/frozen matched-fine-tuning training loop, factored out
after writing the same ~80-line loop 4 times across DeepShip/MIMII/
MIMII-unit/vessel scripts with only the data-loading differing. Takes
plain (waveform, label) lists rather than owning any dataset-specific
manifest/clip logic -- each dataset gets a thin wrapper script that
builds those lists and calls into this module.

Not used to retroactively refactor the earlier per-dataset scripts
(stage5_lora_finetune*.py, the restricted-domain scripts) -- they work,
touching them isn't worth the risk/time right now. This is for FMA-genre/
BirdCLEF/UrbanSound8K and any future dataset.
"""
from __future__ import annotations

import csv
import fcntl
import shutil
from datetime import datetime
from pathlib import Path
from typing import Callable

import audioread.ffdec
import numpy as np
import soundfile as sf
import torch
import torch.nn as nn

from audio_comp.models import get_model_class
from audio_comp.models._util import resample
from audio_comp.pipelines.lora_model_configs import build_lora_model_and_head, lora_prepare_inputs
from audio_comp.pipelines.allora_model_configs import build_allora_model_and_head, allora_prepare_inputs
from audio_comp.pipelines.topk_unfreeze_configs import (
    TOPK_SUPPORTED_MODELS,
    build_topk_model_and_head,
    topk_prepare_inputs,
    topk_trainable_params,
)

EPOCHS = 8
LEARNING_RATE = 1e-4
BATCH_SIZE = 16
EARLY_STOP_PATIENCE = 3
SEED = 0


def write_per_clip_predictions(out_csv: Path, records: list[dict]) -> None:
    """Writes one row per test-set clip: clip_path, true_label, pred_label,
    correct. Added 2026-08-20 to support a complete paired bootstrap of
    Finding 5's gain-side correlation -- the existing clip-level bootstrap
    (project notes) could only resample the *uniformity* side,
    since no fine-tuning run had ever saved individual test predictions,
    only the aggregate accuracy. A plain overwrite (not idempotent-merge
    like write_results_idempotent above): each call is a full rerun for
    one (model, condition, category) triple, so there's no partial-row
    merge concern -- the whole file is this run's complete test set."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["clip_path", "true_label", "pred_label", "correct"])
        writer.writeheader()
        writer.writerows(records)


def write_results_idempotent(
    out_csv: Path, header: list[str], key_cols: list[str], new_rows: list[list]
) -> None:
    """Rewrite-in-place results writer that replaces any existing row
    matching `key_cols` instead of blindly appending -- every finetune_*.py
    script previously used `open(out_csv, "a")`, which meant a rerun (after
    a bugfix, a timeout resubmission, anything) silently left the STALE row
    sitting in the file alongside the corrected one rather than replacing
    it. Found 2026-08-19 auditing Finding 5's UrbanSound8K provenance:
    `results/finetune_urbansound8k.csv` had 223 (model,condition,fold) keys
    with two different accuracy values each -- one from before the
    2026-08-17 sample-rate fix, one from after -- present since the very
    first committed snapshot, not something introduced later. That
    specific case washed out (the correlation happened to have been
    computed with a correct last-occurrence dedup already), but nothing
    guaranteed that, and it silently corrupts any naive `groupby().mean()`
    read of the file. See the project notes 2026-08-19 (cont'd) for the full
    audit. Preserves existing row order for untouched keys; a key that
    already existed gets overwritten in place, a new key is appended.

    Whole read-modify-write critical section is wrapped in an exclusive
    `flock` on a sidecar `.lock` file (added 2026-08-23) -- the row-shape
    validation below catches corruption after the fact, but two Slurm jobs
    finishing within the same second can still interleave their writes to
    the same file no matter how careful the row-parsing is; only a real
    lock prevents that race instead of cleaning up after it. First seen
    2026-08-20 as a fully blank line (`[]`), guarded against then; recurred
    2026-08-23 as a single truncated fragment (`['04645']`, a partial
    accuracy value from an interleaved write) that the blank-line-only
    guard didn't catch -- the row-shape check below is now general
    (any row whose length doesn't match the header is skipped, not just
    empty ones), and the lock addresses the actual root cause so this
    doesn't need a third patch for a third corruption shape."""
    key_idx = [header.index(c) for c in key_cols]
    lock_path = out_csv.with_suffix(out_csv.suffix + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with open(lock_path, "w") as lock_f:
        fcntl.flock(lock_f, fcntl.LOCK_EX)
        try:
            _write_results_idempotent_locked(out_csv, header, key_idx, new_rows)
        finally:
            fcntl.flock(lock_f, fcntl.LOCK_UN)


def _write_results_idempotent_locked(
    out_csv: Path, header: list[str], key_idx: list[int], new_rows: list[list]
) -> None:
    existing: dict[tuple, list[str]] = {}
    order: list[tuple] = []
    if out_csv.exists():
        with open(out_csv, newline="") as f:
            reader = csv.reader(f)
            file_header = next(reader, None)
            if file_header is not None and file_header != header:
                raise ValueError(f"{out_csv} header mismatch: {file_header} != {header}")
            for row in reader:
                if len(row) != len(header):
                    # Malformed row -- either fully blank ([]) or a
                    # truncated fragment from an interleaved concurrent
                    # write (both observed in practice, see this
                    # function's caller's docstring). Skip it; it carries
                    # no recoverable data and would otherwise crash every
                    # subsequent reader of this file with an IndexError.
                    continue
                key = tuple(row[i] for i in key_idx)
                if key not in existing:
                    order.append(key)
                existing[key] = row
    for row in new_rows:
        row = [str(v) for v in row]
        key = tuple(row[i] for i in key_idx)
        if key not in existing:
            order.append(key)
        existing[key] = row
    _snapshot_before_overwrite(out_csv)
    with open(out_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for key in order:
            writer.writerow(existing[key])


def _snapshot_before_overwrite(out_csv: Path) -> None:
    """Copy out_csv's current contents to results/.snapshots/<name>.<date>.csv
    before it gets overwritten in place, at most once per calendar day per
    file. Added 2026-08-21 after Finding 5's retracted 0.4816 discrepancy
    turned out to be genuinely unrecoverable: no commit or backup captured
    the finetune CSVs' exact state when the original check ran, so the
    input behind a since-superseded, already-cited number could not be
    reconstructed, checked, or ruled out as drift -- only concluded
    "unknown." Daily granularity is a deliberate cost/benefit call, not an
    oversight: a full copy on every one of the thousands of per-job writes
    in a multi-seed batch would be wasteful and mostly redundant (jobs land
    continuously, not once), while "what did this file look like on a given
    day" is exactly the resolution this project's own citations operate at
    (e.g. "reconciled 2026-08-20") -- cheap enough to add without becoming
    its own maintenance burden, and it closes the specific failure mode
    that just cost real time to diagnose down to "cause unknown."."""
    if not out_csv.exists():
        return
    snap_dir = out_csv.parent / ".snapshots"
    snap_dir.mkdir(exist_ok=True)
    date_str = datetime.now().strftime("%Y-%m-%d")
    snap_path = snap_dir / f"{out_csv.stem}.{date_str}.csv"
    if not snap_path.exists():
        shutil.copy2(out_csv, snap_path)


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


def read_mono(path: str) -> tuple[np.ndarray, int]:
    """Returns (waveform, TRUE native sample rate of this specific file)
    -- callers must not assume a fixed dataset-wide rate. Caught as a
    real bug 2026-08-17: UrbanSound8K's clips are NOT uniformly 44.1kHz
    as originally assumed (a 200-clip sample showed 44100/48000/96000/
    24000/16000/8000/192000 all present), so passing a single guessed
    constant into embed_batch()/prepare_inputs() as if it were every
    clip's rate would silently resample-or-not incorrectly for the
    majority of non-44.1kHz clips -- the same class of bug already found
    and fixed in stage5_lora_finetune_mimii.py, this time per-clip within
    one dataset rather than across an entire dataset's assumed rate."""
    if path.lower().endswith(".mp3"):
        return _read_mp3_via_ffmpeg(path), _mp3_samplerate_via_ffmpeg(path)
    waveform, sr = sf.read(path, dtype="float32")
    if waveform.ndim > 1:
        waveform = waveform.mean(axis=1)
    return waveform, sr


def _mp3_samplerate_via_ffmpeg(path: str) -> int:
    with audioread.ffdec.FFmpegAudioFile(path) as f:
        return f.samplerate


def read_batch_skip_bad(
    clips: list[dict], target_sample_rate: int, max_duration_s: float | None = None
) -> tuple[list[np.ndarray], list[dict]]:
    """FMA-small ships a handful of genuinely corrupted mp3s (documented
    upstream, e.g. 099134.mp3 -- audio_comp/data/sources/fma.py's own
    probe-set loader already skips these the same way). Returns waveforms
    alongside the SURVIVING clip dicts (not the original `clips` list) so
    labels stay aligned with whichever waveforms actually decoded.

    Every waveform is resampled to `target_sample_rate` here (using each
    file's own true native rate from read_mono(), not an assumed
    dataset-wide constant -- see read_mono()'s docstring), so downstream
    code can treat the returned waveforms as uniformly at one rate, same
    contract as before this fix, just now actually correct per-clip.

    `max_duration_s`, if given, truncates each waveform (in the now-
    common target_sample_rate) after resampling -- added after FMA-
    genre's 30s clips (vs. MIMII/UrbanSound8K/BirdCLEF's 4-10s) OOM'd
    multiple large wav2vec2-family models on a full 80GB GPU during LoRA/
    ALLoRA fine-tuning (real backprop activation memory through a ~1500-
    token sequence at batch=16, not fragmentation alone). X-ARES's own
    upstream fma_genre_config uses crop_length=10 for the same reason --
    matched here rather than inventing a different value."""
    waveforms, kept_clips = [], []
    for c in clips:
        try:
            w, sr = read_mono(c["path"])
            if sr != target_sample_rate:
                w = resample(w, sr, target_sample_rate)
            if max_duration_s is not None:
                max_samples = int(max_duration_s * target_sample_rate)
                w = w[:max_samples]
            waveforms.append(w)
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
    max_duration_s: float | None = None,
    predictions_out_csv: Path | None = None,
    seed: int = SEED,
    return_model: bool = False,
) -> float | tuple[float, "nn.Module", Callable, object]:
    """Shared core for both train_and_eval_lora() and
    train_and_eval_allora() -- identical training loop, only the model
    builder and input-preparation functions differ (LoRA via peft vs.
    ALLoRA via manual module replacement, see allora_model_configs.py).

    `return_model` (added 2026-09-11, purely additive/backward-compatible --
    every existing caller omits it, defaults False, return type unchanged):
    when True, also returns the trained (best-val-state-restored) `trainable`
    module, its `forward_fn`, and `adapter`, as `(test_acc, trainable,
    forward_fn, adapter)` instead of just `test_acc`. Added so a caller can
    do one additional no-grad embedding pass over a different clip set (e.g.
    the full pooled category for a geometry RDM) using the SAME trained
    weights, without training the adapter a second time from scratch.

    `max_duration_s`: see read_batch_skip_bad()'s docstring -- caps clip
    length before it reaches the encoder, added after real OOMs on a full
    80GB GPU with FMA-genre's 30s clips. `torch.cuda.empty_cache()` is
    also called every few batches now (was missing entirely before,
    unlike every other fine-tuning script in this project) to bound
    allocator fragmentation building up across a whole epoch.

    `seed`: threads through model init, training-order shuffling, and
    (via the caller) the train/val split -- added 2026-08-20 to support
    running the same (model, condition, category) multiple times for a
    real per-run variance estimate, not just a single point estimate.
    Defaults to the project's original single-run seed (0) for backward
    compatibility with every prior result already on record."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    trainable, forward_fn, _, embed_dim, adapter = build_fn(model_name, device, num_classes, get_model_class)
    optimizer = torch.optim.AdamW([p for p in trainable.parameters() if p.requires_grad], lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    def run_epoch_eval(clips: list[dict], record_predictions: list | None = None) -> float:
        trainable.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, len(clips), BATCH_SIZE):
                batch = clips[start : start + BATCH_SIZE]
                waveforms, batch = read_batch_skip_bad(batch, native_sample_rate, max_duration_s)
                if not waveforms:
                    continue
                labels = torch.tensor([c["label"] for c in batch], device=device)
                inputs = prepare_inputs_fn(model_name, adapter, waveforms, native_sample_rate, device)
                logits = forward_fn(inputs)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(batch)
                if record_predictions is not None:
                    preds_cpu = preds.detach().cpu().tolist()
                    labels_cpu = labels.detach().cpu().tolist()
                    for clip, true_label, pred_label in zip(batch, labels_cpu, preds_cpu):
                        record_predictions.append({
                            "clip_path": clip.get("path", ""),
                            "true_label": true_label,
                            "pred_label": pred_label,
                            "correct": int(true_label == pred_label),
                        })
                if device == "cuda":
                    torch.cuda.empty_cache()
        return correct / total

    rng = np.random.default_rng(seed)
    best_val_acc, best_state, epochs_without_improvement = -1.0, None, 0
    trainable_param_names = {name for name, p in trainable.named_parameters() if p.requires_grad}

    for epoch in range(epochs):
        trainable.train()
        order = rng.permutation(len(train_clips))
        total_loss = 0.0
        for batch_num, start in enumerate(range(0, len(order), BATCH_SIZE)):
            batch_idx = order[start : start + BATCH_SIZE]
            batch = [train_clips[i] for i in batch_idx]
            waveforms, batch = read_batch_skip_bad(batch, native_sample_rate, max_duration_s)
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
            if device == "cuda" and batch_num % 10 == 0:
                torch.cuda.empty_cache()
        if device == "cuda":
            torch.cuda.empty_cache()
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
    test_predictions: list = [] if predictions_out_csv is not None else None
    test_acc = run_epoch_eval(test_clips, record_predictions=test_predictions)
    if predictions_out_csv is not None:
        write_per_clip_predictions(predictions_out_csv, test_predictions)
    if return_model:
        return test_acc, trainable, forward_fn, adapter
    return test_acc


def train_and_eval_lora(
    model_name: str,
    train_clips: list[dict],
    val_clips: list[dict],
    test_clips: list[dict],
    num_classes: int,
    native_sample_rate: int,
    device: str,
    epochs: int = EPOCHS,
    max_duration_s: float | None = None,
    predictions_out_csv: Path | None = None,
    seed: int = SEED,
) -> float:
    """Each clip dict needs 'path' and 'label' keys. Returns test accuracy
    from the best-val-accuracy epoch (early-stopped). If predictions_out_csv
    is given, also writes one row per test clip (clip_path/true_label/
    pred_label/correct) -- added 2026-08-20 to support a complete paired
    bootstrap of Finding 5's gain-side correlation. `seed` controls model
    init and training-order shuffling (added 2026-08-20 for multi-seed
    variance runs); defaults to 0, the project's original single-run seed."""
    return _train_and_eval_adapter(
        "lora", build_lora_model_and_head, lora_prepare_inputs,
        model_name, train_clips, val_clips, test_clips, num_classes, native_sample_rate, device, epochs,
        max_duration_s, predictions_out_csv, seed,
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
    max_duration_s: float | None = None,
    predictions_out_csv: Path | None = None,
    seed: int = SEED,
) -> float:
    """ALLoRA counterpart to train_and_eval_lora() -- same signature, same
    training loop, only the adapter method differs."""
    return _train_and_eval_adapter(
        "allora", build_allora_model_and_head, allora_prepare_inputs,
        model_name, train_clips, val_clips, test_clips, num_classes, native_sample_rate, device, epochs,
        max_duration_s, predictions_out_csv, seed,
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
    native_sample_rate: int | None = None,
    max_duration_s: float | None = None,
    predictions_out_csv: Path | None = None,
    seed: int = SEED,
) -> float:
    """Frozen-embedding linear probe, matching frozen_baseline_streaming.py's
    own approach (no backbone gradients).

    **Real bug fixed 2026-08-17**: embed_batch() used to be called with
    `adapter.info.expected_sample_rate` as if it were the waveform's true
    native rate -- the same class of bug already caught and fixed in
    stage5_lora_finetune_mimii.py. Harmless when a dataset's native rate
    happens to equal the model's expected rate (coincidentally true for
    most models on 16kHz-native MIMII/UrbanSound8K/BirdCLEF), but a real,
    silent audio-speed corruption for FMA-genre (44.1kHz native) against
    any model expecting a different rate. Now takes the dataset's actual
    native_sample_rate explicitly, same convention as train_and_eval_lora/
    _allora, rather than assuming it matches the model.

    `seed` (added 2026-08-20 for multi-seed variance runs) controls the
    linear head's random init -- the only source of run-to-run randomness
    in this function, since the frozen-embedding probe uses full-batch
    gradient descent with no data shuffling."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    adapter = get_model_class(model_name)(device=device)
    adapter.load()
    embed_sample_rate = native_sample_rate if native_sample_rate is not None else adapter.info.expected_sample_rate

    def embed(clips: list[dict]) -> tuple[np.ndarray, list[dict]]:
        embeddings, kept = [], []
        for start in range(0, len(clips), batch_size):
            batch = clips[start : start + batch_size]
            waveforms, batch = read_batch_skip_bad(batch, embed_sample_rate, max_duration_s)
            if not waveforms:
                continue
            batch_embeds = adapter.embed_batch(waveforms, embed_sample_rate)
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
    if predictions_out_csv is not None:
        head.eval()
        with torch.no_grad():
            preds = head(test_emb).argmax(dim=-1)
        records = [
            {
                "clip_path": clip.get("path", ""),
                "true_label": int(true_label),
                "pred_label": int(pred_label),
                "correct": int(true_label == pred_label),
            }
            for clip, true_label, pred_label in zip(test_clips, test_labels.tolist(), preds.tolist())
        ]
        write_per_clip_predictions(predictions_out_csv, records)
    return eval_acc(test_emb, test_labels)


def train_and_eval_topk(
    model_name: str,
    train_clips: list[dict],
    val_clips: list[dict],
    test_clips: list[dict],
    num_classes: int,
    device: str,
    epochs: int = EPOCHS,
    max_duration_s: float | None = None,
    predictions_out_csv: Path | None = None,
    seed: int = SEED,
) -> float:
    """Top-K-layer-unfreezing counterpart to train_and_eval_lora(), for the
    4 models that can't go through LoRA at all (see
    topk_unfreeze_configs.py's module docstring for why: panns_cnn14/
    audio_jepa/audiomae/musicfm don't use the HF-style self._extractor(...)
    pattern LoRA's shared wiring assumes). Added 2026-08-20 to test whether
    Finding 5's uniformity-vs-adaptation-headroom correlation holds under a
    structurally different, full-rank adaptation mechanism (unfreezing
    whole transformer/conv blocks with plain gradients) or is specific to
    low-rank updates -- a real, previously-untested scope question, since
    this condition's own config module was built 2026-08-16 but never
    actually run against any task before now.

    **Explicitly NOT comparable to the 14-model LoRA/ALLoRA condition and
    must never be blended into the same table** -- per topk_unfreeze_configs.py's
    own standing rule, restated here since this function is the first
    place that condition actually gets exercised end-to-end."""
    if model_name not in TOPK_SUPPORTED_MODELS:
        raise ValueError(f"'{model_name}' not top-K-unfreeze-compatible: {TOPK_SUPPORTED_MODELS}")

    torch.manual_seed(seed)
    np.random.seed(seed)

    base_model, head, forward_fn, sample_rate, embed_dim, adapter = build_topk_model_and_head(
        model_name, device, num_classes, get_model_class
    )
    trainable_params = topk_trainable_params(base_model, head)
    optimizer = torch.optim.AdamW(trainable_params, lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    def run_epoch_eval(clips: list[dict], record_predictions: list | None = None) -> float:
        base_model.eval()
        head.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for start in range(0, len(clips), BATCH_SIZE):
                batch = clips[start : start + BATCH_SIZE]
                waveforms, batch = read_batch_skip_bad(batch, sample_rate, max_duration_s)
                if not waveforms:
                    continue
                labels = torch.tensor([c["label"] for c in batch], device=device)
                inputs = topk_prepare_inputs(model_name, adapter, waveforms, sample_rate, device)
                logits = forward_fn(inputs)
                preds = logits.argmax(dim=-1)
                correct += (preds == labels).sum().item()
                total += len(batch)
                if record_predictions is not None:
                    preds_cpu, labels_cpu = preds.detach().cpu().tolist(), labels.detach().cpu().tolist()
                    for clip, true_label, pred_label in zip(batch, labels_cpu, preds_cpu):
                        record_predictions.append({
                            "clip_path": clip.get("path", ""), "true_label": true_label,
                            "pred_label": pred_label, "correct": int(true_label == pred_label),
                        })
                if device == "cuda":
                    torch.cuda.empty_cache()
        return correct / total

    rng = np.random.default_rng(seed)
    best_val_acc, best_state, epochs_without_improvement = -1.0, None, 0

    for epoch in range(epochs):
        base_model.train()
        head.train()
        order = rng.permutation(len(train_clips))
        total_loss = 0.0
        for batch_num, start in enumerate(range(0, len(order), BATCH_SIZE)):
            batch_idx = order[start : start + BATCH_SIZE]
            batch = [train_clips[i] for i in batch_idx]
            waveforms, batch = read_batch_skip_bad(batch, sample_rate, max_duration_s)
            if not waveforms:
                continue
            labels = torch.tensor([c["label"] for c in batch], device=device)
            inputs = topk_prepare_inputs(model_name, adapter, waveforms, sample_rate, device)
            logits = forward_fn(inputs)
            loss = criterion(logits, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(batch)
            if device == "cuda" and batch_num % 10 == 0:
                torch.cuda.empty_cache()
        if device == "cuda":
            torch.cuda.empty_cache()
        val_acc = run_epoch_eval(val_clips)
        print(f"  [topk:{model_name}] epoch {epoch + 1}/{epochs} loss={total_loss / len(train_clips):.4f} val_acc={val_acc:.4f}", flush=True)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            # Whole state_dict is checkpointed (not just the unfrozen
            # blocks) -- simpler and equally correct, since frozen params
            # never change: reloading the full dict is a no-op for them.
            best_state = {
                "model": {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()},
                "head": {k: v.detach().cpu().clone() for k, v in head.state_dict().items()},
            }
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= EARLY_STOP_PATIENCE:
                print(f"  [topk:{model_name}] early stopping at epoch {epoch + 1} (best val_acc={best_val_acc:.4f})")
                break

    base_model.load_state_dict({k: v.to(device) for k, v in best_state["model"].items()}, strict=False)
    head.load_state_dict({k: v.to(device) for k, v in best_state["head"].items()})

    test_predictions: list = [] if predictions_out_csv is not None else None
    test_acc = run_epoch_eval(test_clips, record_predictions=test_predictions)
    if predictions_out_csv is not None:
        write_per_clip_predictions(predictions_out_csv, test_predictions)
    return test_acc
