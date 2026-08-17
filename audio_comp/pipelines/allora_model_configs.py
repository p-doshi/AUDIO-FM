"""Per-model ALLoRA wiring -- mirrors lora_model_configs.py's structure
exactly (same 14-model roster, same target_modules per model, same
per-model prepare_inputs/forward functions, same excluded-model list and
reasons) but replaces peft's `LoraConfig`/`get_peft_model()` with manual
in-place replacement of the target nn.Linear submodules by
`allora.ALLoRALinear`, since ALLoRA has no peft integration (it's a
different gradient rule, not expressible as a peft adapter config).

r=8 to match the existing LoRA runs' rank exactly, for a genuinely
matched before/after comparison -- ALLoRA introduces no new
hyperparameter beyond r and eta (eta replaces LoRA's alpha/r scaling
conceptually, but is NOT applied as a forward-pass scale -- see
allora.py's module docstring). eta=8 follows the ALLoRA paper's own
Table 1 setup (eta^2 in {1,2,4} tested there; eta=8 i.e. eta^2=64 is
outside that swept range -- chosen instead to match this project's
existing LORA_ALPHA=16, LORA_RANK=8 pair, i.e. treating eta similarly to
how alpha was chosen relative to r previously, not because the paper
specifically validates this value. Worth a small eta sweep later if
initial results are sensitive to it -- not done this pass, matched
protocol takes priority over hyperparameter tuning for a first
comparison.
"""
from __future__ import annotations

import re
from typing import Callable

import numpy as np
import torch
import torch.nn as nn

from audio_comp.pipelines.allora import ALLoRALinear
from audio_comp.pipelines.lora_model_configs import (
    LORA_EXCLUDED_MODELS,
    MODEL_LORA_SPECS,
)

ALLORA_RANK = 8
ALLORA_ETA = 8.0

ALLORA_SUPPORTED_MODELS = sorted(MODEL_LORA_SPECS.keys())
ALLORA_EXCLUDED_MODELS = dict(LORA_EXCLUDED_MODELS)  # identical exclusion set/reasons as LoRA


def _leaf_name(dotted_path: str) -> str:
    return dotted_path.rsplit(".", 1)[-1]


def apply_allora(model: nn.Module, target_modules, r: int, eta: float) -> list[ALLoRALinear]:
    """In-place replacement of every nn.Linear submodule matching
    `target_modules` (a list of leaf names, matching peft's own default
    matching convention, OR a regex string matched against the full
    dotted path -- same two forms lora_model_configs.py's
    MODEL_LORA_SPECS already uses for clap's audio-tower-only scoping)."""
    is_regex = isinstance(target_modules, str)
    pattern = re.compile(target_modules) if is_regex else None

    replaced = []
    # Collect matches first -- mutating named_modules() while iterating it is unsafe.
    matches = []
    for dotted_path, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if is_regex:
            if pattern.search(dotted_path):
                matches.append(dotted_path)
        else:
            if _leaf_name(dotted_path) in target_modules:
                matches.append(dotted_path)

    for dotted_path in matches:
        parent_path, _, attr = dotted_path.rpartition(".")
        parent = model.get_submodule(parent_path) if parent_path else model
        original = getattr(parent, attr)
        wrapped = ALLoRALinear(original, r=r, eta=eta)
        setattr(parent, attr, wrapped)
        replaced.append(wrapped)

    return replaced


def build_allora_model_and_head(
    model_name: str, device: str, num_classes: int, get_model_class
) -> tuple[nn.Module, Callable, int, int, object]:
    """Mirrors lora_model_configs.build_lora_model_and_head()'s return
    signature and calling convention exactly, so the same training-loop
    code (generic_lora_trainer.py etc.) can be reused for either method
    with only the builder function swapped."""
    spec = MODEL_LORA_SPECS[model_name]
    adapter = get_model_class(model_name)(device=device)
    adapter.load()

    # Freeze the ENTIRE base model first -- apply_allora only replaces the
    # target Linear submodules (whose own base_layer it separately freezes
    # in ALLoRALinear.__init__), it does not touch anything else. Without
    # this, every non-target parameter (embeddings, layer norms, untargeted
    # linears, ...) stays trainable by default -- caught via the smoke
    # test's own trainable_params count coming back as the full model size
    # (tens to hundreds of millions) instead of a LoRA-sized number.
    if isinstance(adapter._model, nn.Module):
        for p in adapter._model.parameters():
            p.requires_grad = False

    replaced_layers = apply_allora(adapter._model, spec.target_modules, ALLORA_RANK, ALLORA_ETA)
    if not replaced_layers:
        raise RuntimeError(f"apply_allora matched zero Linear layers for '{model_name}' -- target_modules wrong?")
    for layer in replaced_layers:
        layer.to(device)

    with torch.no_grad():
        dummy_inputs = spec.prepare_inputs(adapter, [np.zeros(16000, dtype="float32")], 16000, device)
        embed_dim = spec.forward(adapter._model, dummy_inputs).shape[-1]

    head = nn.Linear(embed_dim, num_classes).to(device)
    trainable = nn.ModuleDict(
        {
            "model": adapter._model if isinstance(adapter._model, nn.Module) else nn.Identity(),
            "head": head,
        }
    )
    # adapter._model's frozen base params + our replaced ALLoRALinear's
    # base_layer are already requires_grad=False (set in ALLoRALinear's
    # __init__); only lora_A/lora_B (registered as nn.Parameter directly
    # on each ALLoRALinear, hence visible via adapter._model.parameters())
    # and the head end up with requires_grad=True.

    def forward_fn(inputs):
        pooled = spec.forward(adapter._model, inputs)
        return head(pooled)

    return trainable, forward_fn, adapter.info.expected_sample_rate, embed_dim, adapter


def allora_prepare_inputs(model_name: str, adapter, waveforms, sample_rate: int, device: str) -> dict:
    return MODEL_LORA_SPECS[model_name].prepare_inputs(adapter, waveforms, sample_rate, device)
