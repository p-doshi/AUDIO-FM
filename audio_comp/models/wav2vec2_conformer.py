"""wav2vec2-Conformer — wav2vec2's masked-prediction objective with a
Conformer (convolution-augmented transformer) encoder instead of a plain
transformer, the closest within-roster test of architecture (not
paradigm/domain) as a geometry-shaping axis alongside PANNs' pure-CNN
comparison (the project plan Stage 3 interim finding: architecture family was not
supported as an independent axis there -- this is a second, more targeted
test, holding paradigm AND training data fixed to plain wav2vec2's own
960h LibriSpeech, varying only the encoder block design). Stage 2 addition
(2026-08-12), part of the push toward ~20 models. License verified
directly against the HF model card: Apache-2.0. Self-supervised base
checkpoint (`rel-pos-large`), not fine-tuned.
"""
from __future__ import annotations

import os
from typing import Sequence

import numpy as np
import torch
from transformers import Wav2Vec2ConformerModel, Wav2Vec2FeatureExtractor

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample


def _multi_gpu_max_memory(priority_device: int, other_device: int, safety_margin_gib: float | None = None) -> dict:
    """Split the model across two GPUs via `transformers`/`accelerate`'s
    device_map="auto", filling `priority_device` first and only spilling
    onto `other_device` (and CPU as a last resort) once it's full --
    added 2026-08-23 for the wav2vec2_conformer OOM-under-contention case
    (see the project notes), where a third-party process was holding ~85GiB on
    one GPU leaving <1GiB free there, but a second GPU had real headroom.
    Reads CURRENT free memory at call time (not total capacity) so it
    naturally works around whatever another process is already holding,
    rather than assuming exclusive access to either device.

    `safety_margin_gib` default raised from 1.0 to 4.0 (2026-08-23, second
    incident same day): the 1GiB margin only covered *weight* placement,
    with no headroom for forward-pass activation memory, and on a
    genuinely contended GPU (other jobs actively growing their own
    footprint between this function's one-shot mem_get_info() call and
    the actual forward pass) that 1GiB evaporated before the model even
    ran -- observed directly: budget said 5.89GiB free on GPU0 at load
    time, only 1.65GiB was actually free by forward time, two other
    processes on the same device having grown to 42.30GiB each in the
    interim. This margin can't eliminate that race (nothing short of an
    OS-level reservation can), but a wider buffer plus embed_batch's own
    OOM-triggered retry (below) makes a transient contention spike
    survivable instead of fatal. Override via
    WAV2VEC2_CONFORMER_SAFETY_MARGIN_GIB if 4.0 still isn't enough."""
    if safety_margin_gib is None:
        safety_margin_gib = float(os.environ.get("WAV2VEC2_CONFORMER_SAFETY_MARGIN_GIB", "4.0"))
    free_priority, _ = torch.cuda.mem_get_info(priority_device)
    free_other, _ = torch.cuda.mem_get_info(other_device)
    margin = int(safety_margin_gib * 1024 ** 3)
    budget_priority = max(0, free_priority - margin)
    budget_other = max(0, free_other - margin)
    return {
        priority_device: f"{budget_priority / 1024**3:.2f}GiB",
        other_device: f"{budget_other / 1024**3:.2f}GiB",
        "cpu": "64GiB",  # last-resort overflow if both GPUs combined still aren't enough
    }


@register_model("wav2vec2_conformer")
class Wav2Vec2ConformerEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="wav2vec2_conformer",
        hf_id="facebook/wav2vec2-conformer-rel-pos-large",
        paradigm="masked modeling (speech, Conformer encoder architecture)",
        license="Apache-2.0",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(self.info.hf_id)

        # Opt-in multi-GPU sharding, gated by env var so every OTHER
        # model/script keeps its normal single-.to(device) behavior --
        # this is a targeted workaround for one contended-GPU incident,
        # not a change to the project's general loading path. Set
        # WAV2VEC2_CONFORMER_MULTI_GPU=1 (and optionally
        # WAV2VEC2_CONFORMER_PRIORITY_GPU=<int>, WAV2VEC2_CONFORMER_OTHER_GPU=<int>,
        # default 1 and 0 respectively -- GPU 1 first since GPU 0 already
        # has this project's own "OG mass run" on it) to enable.
        if os.environ.get("WAV2VEC2_CONFORMER_MULTI_GPU") == "1":
            priority_gpu = int(os.environ.get("WAV2VEC2_CONFORMER_PRIORITY_GPU", "1"))
            other_gpu = int(os.environ.get("WAV2VEC2_CONFORMER_OTHER_GPU", "0"))
            max_memory = _multi_gpu_max_memory(priority_gpu, other_gpu)
            print(f"[wav2vec2_conformer] multi-GPU device_map enabled, max_memory={max_memory}", flush=True)
            self._model = Wav2Vec2ConformerModel.from_pretrained(
                self.info.hf_id, device_map="auto", max_memory=max_memory
            ).eval()
            # embed_batch() below moves inputs to `self.device` -- with a
            # sharded model that must be wherever the FIRST layer landed,
            # not necessarily `priority_gpu` (e.g. if priority_gpu had
            # zero budget after the safety margin, accelerate starts on
            # other_gpu instead). Read it back from the dispatched model
            # rather than assuming.
            self.device = next(self._model.parameters()).device
        else:
            self._model = Wav2Vec2ConformerModel.from_pretrained(self.info.hf_id).to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        # OOM-triggered retry, added 2026-08-23 alongside the wider safety
        # margin above -- on a contended GPU, a batch that would have fit
        # against the memory budget measured at model-load time can still
        # OOM later if a neighboring process grew in between (observed
        # directly the same day: budget said 5.89GiB free, only 1.65GiB
        # was left by forward time). Halving the batch and retrying after
        # clearing the cache survives a transient spike instead of killing
        # the whole run; single-clip batches still failing is treated as a
        # real, non-transient OOM and re-raised.
        if len(waveforms) > 1:
            try:
                return self._embed_batch_once(waveforms, sample_rate)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                mid = len(waveforms) // 2
                print(f"[wav2vec2_conformer] OOM on batch of {len(waveforms)}, "
                      f"retrying as two batches of {mid}/{len(waveforms) - mid}", flush=True)
                first = self.embed_batch(waveforms[:mid], sample_rate)
                second = self.embed_batch(waveforms[mid:], sample_rate)
                return np.concatenate([first, second], axis=0)
        return self._embed_batch_once(waveforms, sample_rate)

    def _embed_batch_once(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        inputs = self._extractor(
            resampled, sampling_rate=self.info.expected_sample_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = self._model(**inputs)
        return mean_pool(outputs.last_hidden_state).cpu().numpy()
