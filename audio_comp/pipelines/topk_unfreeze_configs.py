"""Top-K-layer unfreezing, a SECOND, separately-reported fine-tuning
condition for the 4 models that cannot go through LoRA at all
(see lora_model_configs.py's module docstring for why: none of
panns_cnn14/audio_jepa/audiomae/musicfm use the HF-style
self._extractor(...) pattern LoRA's shared wiring assumes; panns_cnn14
also has no attention layer for LoRA to target regardless).

**This is explicitly NOT comparable to the 14-model LoRA condition and
must never be blended into the same comparison table or correlation** --
per the user's own explicit direction (2026-08-16): report it as its own
labeled condition, the same pattern CLAUDE.md already uses for "official
fine-tuned" vs. "matched" conditions. Unfreezing the last K full
transformer/conv blocks and training them with plain gradients is a
materially different, generally stronger adaptation technique than
rank-8 LoRA on just attention projections -- mixing the two into one
"fine-tuned" column would silently conflate adaptation-method strength
with representation quality, exactly the confound this project's Stage 5
methodology exists to avoid.

`encodecmae` is excluded even from this condition -- its
extract_features_from_array() convenience method is genuinely
non-differentiable as written (wraps everything in torch.no_grad(),
round-trips through numpy mid-pipeline via self.apply_processors(), see
the 2026-08-16 journal entry) -- building a real gradient path through it
would mean reimplementing its internal chunking/activation-extraction
logic from scratch, real unbounded work, not attempted this pass.
`encodecmae` still participates in the frozen-probe condition normally.

Per-model unfrozen-block config (verified 2026-08-16 via named_children()):
  - panns_cnn14: conv_block1..6 (6 conv blocks) + fc1 -- unfreeze last
    TOPK_BLOCKS conv blocks plus fc1 (the projection into the 2048-d
    embedding space that's actually used).
  - audio_jepa: .blocks (12-block ModuleList, ViT-style) + .norm --
    unfreeze last TOPK_BLOCKS transformer blocks plus the final norm.
  - audiomae: .blocks (12-block ModuleList, MAE encoder -- .decoder_blocks
    is the reconstruction decoder, never used for embeddings, left
    frozen/untouched) + .norm.
  - musicfm: .conformer.layers (12-layer ModuleList) -- unfreeze last
    TOPK_BLOCKS conformer layers. get_latent(layer_ix=7) reads an
    intermediate layer's output, so unfreezing the *last* K layers only
    affects the embedding if layer_ix falls within the unfrozen range --
    stated explicitly since it's a real, non-obvious interaction, not a
    generic "just unfreeze the end" assumption. With TOPK_BLOCKS=2
    (layers 10-11) and layer_ix=7, the extracted layer's own weights stay
    frozen but earlier gradient flow doesn't reach it either way; this is
    a known limitation, not silently glossed over -- musicfm's contribution
    under top-K unfreeze may behave more like the frozen condition than
    the other three unless TOPK_BLOCKS is raised past 12-7=5.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from audio_comp.models._util import mean_pool, resample

TOPK_BLOCKS = 2  # matched across all 4 models in this condition

TOPK_EXCLUDED_MODELS = {
    "encodecmae": "extract_features_from_array() is non-differentiable as written "
    "(torch.no_grad() + numpy round-trips mid-pipeline) -- would need internal reimplementation",
}


@dataclass
class TopKModelSpec:
    unfreeze: Callable[[nn.Module], list[nn.Parameter]]  # returns the params to unfreeze
    prepare_inputs: Callable
    forward: Callable  # (model, inputs) -> pooled embedding tensor (batch, dim)
    sample_rate_attr: str = "expected_sample_rate"


def _unfreeze_panns(model: nn.Module) -> list[nn.Parameter]:
    params = []
    for block_name in [f"conv_block{i}" for i in range(6 - TOPK_BLOCKS + 1, 7)] + ["fc1"]:
        for p in getattr(model, block_name).parameters():
            p.requires_grad = True
            params.append(p)
    return params


def _panns_prepare_inputs(adapter, waveforms, sample_rate, device):
    resampled = [resample(w, sample_rate, adapter.info.expected_sample_rate) for w in waveforms]
    min_samples = int(adapter.info.expected_sample_rate * adapter.MIN_SAFE_SAMPLES_S)
    max_len = max(min_samples, max(len(w) for w in resampled))
    batch = np.stack([np.pad(w, (0, max_len - len(w))) for w in resampled]).astype(np.float32)
    return {"wav": torch.from_numpy(batch).to(device)}


def _panns_forward(model, inputs):
    return model(inputs["wav"])["embedding"]


def _unfreeze_vit_blocks(model: nn.Module) -> list[nn.Parameter]:
    params = []
    for block in model.blocks[-TOPK_BLOCKS:]:
        for p in block.parameters():
            p.requires_grad = True
            params.append(p)
    for p in model.norm.parameters():
        p.requires_grad = True
        params.append(p)
    return params


def _audio_jepa_prepare_inputs(adapter, waveforms, sample_rate, device):
    from audio_comp.models.audio_jepa import CLIP_LENGTH_S, _compute_mel_spec

    min_samples = adapter.info.expected_sample_rate * CLIP_LENGTH_S
    specs = []
    for waveform in waveforms:
        resampled = resample(waveform, sample_rate, adapter.info.expected_sample_rate)
        if len(resampled) < min_samples:
            resampled = np.pad(resampled, (0, min_samples - len(resampled)))
        wav_tensor = torch.from_numpy(resampled).float().unsqueeze(0)
        specs.append(_compute_mel_spec(wav_tensor))
    return {"batch": torch.stack(specs).to(device)}


def _audio_jepa_forward(model, inputs):
    return model(inputs["batch"]).mean(dim=1)


def _audiomae_prepare_inputs(adapter, waveforms, sample_rate, device):
    from audio_comp.models.audiomae import _compute_fbank

    resampled = [resample(w, sample_rate, adapter.info.expected_sample_rate) for w in waveforms]
    fbanks = [_compute_fbank(torch.from_numpy(w).unsqueeze(0), adapter.info.expected_sample_rate) for w in resampled]
    return {"batch": torch.stack(fbanks).to(device)}


def _audiomae_forward(model, inputs):
    return mean_pool(model.forward_encoder_no_mask(inputs["batch"]))


def _unfreeze_musicfm(model: nn.Module) -> list[nn.Parameter]:
    params = []
    for layer in model.conformer.layers[-TOPK_BLOCKS:]:
        for p in layer.parameters():
            p.requires_grad = True
            params.append(p)
    return params


MUSICFM_MIN_SAFE_SAMPLES_S = 1.0  # matches panns_cnn14's own short-clip safety margin convention


def _musicfm_prepare_inputs(adapter, waveforms, sample_rate, device):
    # Caught by the mandatory standalone-60ms-clip smoke test 2026-08-16:
    # musicfm's internal preprocessing needs padding >= 1024 samples at
    # some conv/pool stage, which a 960-sample (60ms @ 16kHz) clip can't
    # satisfy -- the same short-clip lesson this project has hit with
    # audio_jepa/panns_cnn14 before. Pad to a safe minimum first, same
    # pattern as panns_cnn14.py's MIN_SAFE_SAMPLES_S. Worth checking
    # whether musicfm.py's own production embed_batch() needs the same
    # fix -- not yet hit in practice (probe-set/vessel clips have all
    # been long enough so far) but the same latent bug exists there.
    resampled = [resample(w, sample_rate, adapter.info.expected_sample_rate) for w in waveforms]
    min_samples = int(adapter.info.expected_sample_rate * MUSICFM_MIN_SAFE_SAMPLES_S)
    max_len = max(min_samples, max(len(w) for w in resampled))
    batch = np.stack([np.pad(w, (0, max_len - len(w))) for w in resampled]).astype(np.float32)
    return {"wav": torch.from_numpy(batch).to(device)}


def _musicfm_forward(model, inputs):
    return mean_pool(model.get_latent(inputs["wav"], layer_ix=7))


TOPK_MODEL_SPECS: dict[str, TopKModelSpec] = {
    "panns_cnn14": TopKModelSpec(unfreeze=_unfreeze_panns, prepare_inputs=_panns_prepare_inputs, forward=_panns_forward),
    "audio_jepa": TopKModelSpec(
        unfreeze=_unfreeze_vit_blocks, prepare_inputs=_audio_jepa_prepare_inputs, forward=_audio_jepa_forward
    ),
    "audiomae": TopKModelSpec(
        unfreeze=_unfreeze_vit_blocks, prepare_inputs=_audiomae_prepare_inputs, forward=_audiomae_forward
    ),
    "musicfm": TopKModelSpec(unfreeze=_unfreeze_musicfm, prepare_inputs=_musicfm_prepare_inputs, forward=_musicfm_forward),
}
TOPK_SUPPORTED_MODELS = sorted(TOPK_MODEL_SPECS.keys())


def build_topk_model_and_head(
    model_name: str, device: str, num_classes: int, get_model_class
) -> tuple[nn.Module, nn.Module, Callable, int, int, object]:
    """Returns (base_model, head, forward_fn(inputs)->logits, sample_rate, embed_dim, adapter).

    Unlike LoRA, no peft wrapping -- the base model's own last-K-block
    params get requires_grad=True directly, everything else stays frozen.
    """
    spec = TOPK_MODEL_SPECS[model_name]
    adapter = get_model_class(model_name)(device=device)
    adapter.load()
    model = adapter._model

    for p in model.parameters():
        p.requires_grad = False
    spec.unfreeze(model)

    with torch.no_grad():
        dummy_inputs = spec.prepare_inputs(adapter, [np.zeros(16000, dtype="float32")], 16000, device)
        embed_dim = spec.forward(model, dummy_inputs).shape[-1]

    head = nn.Linear(embed_dim, num_classes).to(device)

    def forward_fn(inputs):
        pooled = spec.forward(model, inputs)
        return head(pooled)

    return model, head, forward_fn, adapter.info.expected_sample_rate, embed_dim, adapter


def topk_prepare_inputs(model_name: str, adapter, waveforms, sample_rate: int, device: str) -> dict:
    return TOPK_MODEL_SPECS[model_name].prepare_inputs(adapter, waveforms, sample_rate, device)


def topk_trainable_params(model: nn.Module, head: nn.Module) -> list[nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad] + list(head.parameters())
