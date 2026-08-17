"""Per-model LoRA target-module configs and input/forward wiring, shared
by stage5_lora_finetune.py (DeepShip), stage5_lora_finetune_mimii.py, and
data/AIS's vessel-classification scripts.

Extended 2026-08-16 from the original 5-model set (wav2vec2/hubert/mert/
music2vec/ast, all sharing q_proj/v_proj naming) after directly inspecting
named_modules() for every model in configs/models.yaml's active_models
(19 models). Verified against actual module names, not assumed from
architecture family -- same standing rule this project applies to every
checkpoint/license claim.

Coverage after this pass: 14 of 19 active models support LoRA through this
shared wiring. Five do NOT, for stated architectural reasons -- silently
dropping them would misrepresent "all models" as fully achieved when it
isn't:
  - `panns_cnn14`: pure CNN, genuinely no attention layer for LoRA
    (query/key/value-style) to target. Would need a fundamentally
    different adaptation technique (e.g. LoRA on Conv2d layers), not a
    naming difference -- out of scope for this pass.
  - `audio_jepa`, `audiomae`, `musicfm`, `encodecmae`: none of these use
    an HF-style `self._extractor(...)` + `self._model(**inputs)` call --
    each has bespoke preprocessing baked directly into embed_batch()
    (kaldi.fbank features, ViT patchify, a custom `get_latent()` API, a
    codec-specific loader). Making these differentiable end-to-end for
    LoRA needs individually reading and reimplementing each one's actual
    preprocessing path, not a target_modules addition -- real, bounded
    follow-up work, not attempted this pass.

Per-model target_modules groups (verified 2026-08-16 via named_modules()):
  - q_proj/v_proj (standard HF wav2vec2-style attention): wav2vec2,
    hubert, mert, music2vec, ast, data2vec_audio, mms, sew,
    unispeech_sat, wavlm, whisper (11 models)
  - linear_q/linear_v (Conformer relative-position attention, distinct
    naming from plain wav2vec2 despite same training data/objective):
    wav2vec2_conformer (1 model)
  - qkv (fused ViT-style attention): bird_mae (1 model; audiomae and
    audio_jepa also use fused qkv but are excluded above for the
    preprocessing reason, not the target_modules reason)
  - query/value, audio-tower only (CLAP has both an audio_model and a
    text_model; a bare "query"/"value" leaf-name match would inject LoRA
    into the unused text tower too -- wasteful trainable params, not a
    correctness bug, but avoided via a regex scoped to `audio_model.*`):
    clap (1 model)

Total: 11 + 1 + 1 + 1 = 14 models.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

from audio_comp.models._util import mean_pool, resample

LORA_RANK = 8
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

WAV2VEC2_STYLE_QV = [
    "wav2vec2", "hubert", "mert", "music2vec", "ast",
    "data2vec_audio", "mms", "sew", "unispeech_sat", "wavlm", "whisper",
]


@dataclass
class LoraModelSpec:
    target_modules: list[str] | str
    prepare_inputs: Callable
    forward: Callable  # (peft_model, inputs) -> pooled embedding tensor (batch, dim)


def _standard_prepare_inputs(adapter, waveforms, sample_rate, device):
    """wav2vec2-family pattern: self._extractor(..., padding=True) -> dict
    with input_values, moved to device."""
    resampled = [resample(w, sample_rate, adapter.info.expected_sample_rate) for w in waveforms]
    inputs = adapter._extractor(
        resampled, sampling_rate=adapter.info.expected_sample_rate, return_tensors="pt", padding=True
    )
    return {k: v.to(device) for k, v in inputs.items()}


def _standard_forward(peft_model, inputs):
    return mean_pool(peft_model(**inputs).last_hidden_state)


def _ast_forward(peft_model, inputs):
    return peft_model(**inputs).pooler_output


def _whisper_prepare_inputs(adapter, waveforms, sample_rate, device):
    resampled = [resample(w, sample_rate, adapter.info.expected_sample_rate) for w in waveforms]
    inputs = adapter._extractor(resampled, sampling_rate=adapter.info.expected_sample_rate, return_tensors="pt")
    return {"input_features": inputs["input_features"].to(device)}


def _whisper_forward(model, inputs):
    # For peft-wrapped LoRA, `model` is a PeftModel wrapping the original
    # WhisperModel at `.base_model.model`; its `.encoder` submodule is
    # where LoRA was injected and where the actual forward pass must go --
    # calling model(**inputs) directly would invoke the full
    # encoder-decoder, which this project never uses. For ALLoRA (no peft
    # involved, module replacement happens directly on the raw model),
    # `model` already IS the WhisperModel. Checking `hasattr(model,
    # "base_model")` alone is NOT a reliable peft-vs-raw test: HF's own
    # PreTrainedModel defines a `base_model` *property* too (returns self
    # for a bare WhisperModel with no task head) -- caught via a real
    # AttributeError during the ALLoRA smoke test, not assumed. Check for
    # `peft_config` instead, an attribute only a real peft.PeftModel has.
    whisper_model = model.base_model.model if hasattr(model, "peft_config") else model
    encoder_out = whisper_model.encoder(inputs["input_features"])
    return mean_pool(encoder_out.last_hidden_state)


def _bird_mae_prepare_inputs(adapter, waveforms, sample_rate, device):
    resampled = [resample(w, sample_rate, adapter.info.expected_sample_rate) for w in waveforms]
    max_len = max(len(w) for w in resampled)
    import numpy as np

    batch = np.stack([np.pad(w, (0, max_len - len(w))) for w in resampled]).astype("float32")
    features = adapter._extractor(batch, return_tensors="pt").to(device)
    return {"input_values": features}


def _bird_mae_forward(peft_model, inputs):
    # Unlike the wav2vec2-style models, bird_mae's last_hidden_state comes
    # out already pooled to (batch, hidden) -- confirmed 2026-08-16 by
    # direct inspection (torch.Size([1, 768]) for a 1-clip batch, not
    # (1, seq, 768)) -- applying mean_pool() on top would incorrectly
    # average over the hidden dimension instead of a (nonexistent) time
    # axis, silently producing garbage instead of erroring loudly (it did
    # error here, via a downstream shape mismatch, but only by luck).
    return peft_model(**inputs).last_hidden_state


def _clap_prepare_inputs(adapter, waveforms, sample_rate, device):
    resampled = [resample(w, sample_rate, adapter.info.expected_sample_rate) for w in waveforms]
    inputs = adapter._processor(audio=resampled, sampling_rate=adapter.info.expected_sample_rate, return_tensors="pt")
    return {k: v.to(device) for k, v in inputs.items()}


def _clap_forward(peft_model, inputs):
    # get_audio_features() never touches text_model, so text-tower LoRA
    # params (excluded via target_modules regex below anyway) would get
    # zero gradient even if included -- this call path only exercises
    # audio_model, matching why target_modules is scoped to it.
    return peft_model.get_audio_features(**inputs).pooler_output


MODEL_LORA_SPECS: dict[str, LoraModelSpec] = {}
for _name in WAV2VEC2_STYLE_QV:
    MODEL_LORA_SPECS[_name] = LoraModelSpec(
        target_modules=["q_proj", "v_proj"],
        prepare_inputs=(_whisper_prepare_inputs if _name == "whisper" else _standard_prepare_inputs),
        forward=(
            _ast_forward if _name == "ast" else _whisper_forward if _name == "whisper" else _standard_forward
        ),
    )
MODEL_LORA_SPECS["wav2vec2_conformer"] = LoraModelSpec(
    target_modules=["linear_q", "linear_v"],
    prepare_inputs=_standard_prepare_inputs,
    forward=_standard_forward,
)
MODEL_LORA_SPECS["bird_mae"] = LoraModelSpec(
    target_modules=["qkv"],
    prepare_inputs=_bird_mae_prepare_inputs,
    forward=_bird_mae_forward,
)
MODEL_LORA_SPECS["clap"] = LoraModelSpec(
    target_modules=r"audio_model.*\.(query|value)$",
    prepare_inputs=_clap_prepare_inputs,
    forward=_clap_forward,
)

LORA_SUPPORTED_MODELS = sorted(MODEL_LORA_SPECS.keys())

# Documented, not silently dropped -- see module docstring for why each is excluded.
LORA_EXCLUDED_MODELS = {
    "panns_cnn14": "pure CNN, no attention layer for LoRA to target",
    "audio_jepa": "custom (non-HF-extractor) preprocessing, needs bespoke input wiring",
    "audiomae": "custom (non-HF-extractor) preprocessing, needs bespoke input wiring",
    "musicfm": "custom (non-HF-extractor) preprocessing, needs bespoke input wiring",
    "encodecmae": "custom (non-HF-extractor) preprocessing, needs bespoke input wiring",
}


def build_lora_model_and_head(
    model_name: str, device: str, num_classes: int, get_model_class
) -> tuple[nn.Module, Callable, int, int, object]:
    """Returns (trainable, forward_fn(inputs)->logits, sample_rate, embed_dim, adapter).

    Mirrors stage5_lora_finetune.py's build_model_and_head, generalized to
    any model in MODEL_LORA_SPECS via its registered target_modules/
    forward wiring instead of a single hardcoded config.
    """
    spec = MODEL_LORA_SPECS[model_name]
    adapter = get_model_class(model_name)(device=device)
    adapter.load()

    lora_config = LoraConfig(
        r=LORA_RANK,
        lora_alpha=LORA_ALPHA,
        target_modules=spec.target_modules,
        lora_dropout=LORA_DROPOUT,
        bias="none",
    )
    peft_model = get_peft_model(adapter._model, lora_config).to(device)

    # Probe embed_dim once via a tiny dummy forward rather than assuming a
    # config attribute name (varies across architectures: hidden_size,
    # embed_dim, projection_dim, etc.) -- robust the same way regardless
    # of which of the 14 models this is.
    import numpy as np

    with torch.no_grad():
        dummy_inputs = spec.prepare_inputs(adapter, [np.zeros(16000, dtype="float32")], 16000, device)
        embed_dim = spec.forward(peft_model, dummy_inputs).shape[-1]

    trainable = nn.ModuleDict({"peft_model": peft_model, "head": nn.Linear(embed_dim, num_classes).to(device)})

    def forward_fn(inputs):
        pooled = spec.forward(trainable["peft_model"], inputs)
        return trainable["head"](pooled)

    return trainable, forward_fn, adapter.info.expected_sample_rate, embed_dim, adapter


def lora_prepare_inputs(model_name: str, adapter, waveforms, sample_rate: int, device: str) -> dict:
    return MODEL_LORA_SPECS[model_name].prepare_inputs(adapter, waveforms, sample_rate, device)
