"""Model registry: string name -> BaseAudioEncoder subclass.

Two separate checks guard checkpoint provenance (see base.py's
CHECKPOINT_STATUSES / COMPARISON_ELIGIBLE_CHECKPOINT_STATUSES for the
full rationale):
  - register_model() validates every model declares a *known* status at
    import time — catches a missing/misspelled checkpoint_status
    immediately, before the model is ever used for anything.
  - get_model_class() additionally validates the status is *comparison-
    eligible* — this is the actual "excluded from the main RSA/CKA
    comparison" enforcement, since every pipeline entry point
    (extract_embeddings.py, xares_eval's encoders) resolves a model name
    through this function before instantiating it. A community_conversion
    or code_only model can still be registered (so the framework already
    models it, same as `beats` sitting in deferred_models before its
    loader existed) — it just can't be resolved for actual use until its
    checkpoint_status is upgraded.
"""
from __future__ import annotations

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseAudioEncoder

MODEL_REGISTRY: Dict[str, Type["BaseAudioEncoder"]] = {}


def register_model(name: str):
    def _decorator(cls):
        from .base import CHECKPOINT_STATUSES

        if name in MODEL_REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        status = getattr(cls.info, "checkpoint_status", None)
        if status not in CHECKPOINT_STATUSES:
            raise ValueError(
                f"model '{name}': checkpoint_status={status!r} is not one of {CHECKPOINT_STATUSES}"
            )
        MODEL_REGISTRY[name] = cls
        return cls

    return _decorator


def get_model_class(name: str) -> Type["BaseAudioEncoder"]:
    from .base import COMPARISON_ELIGIBLE_CHECKPOINT_STATUSES

    try:
        cls = MODEL_REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"no model registered as '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        ) from e

    status = cls.info.checkpoint_status
    if status not in COMPARISON_ELIGIBLE_CHECKPOINT_STATUSES:
        raise ValueError(
            f"model '{name}' has checkpoint_status={status!r}, which is not eligible for the "
            f"main RSA/CKA comparison (needs one of {sorted(COMPARISON_ELIGIBLE_CHECKPOINT_STATUSES)}). "
            "Upgrade its checkpoint_status (with verification against the model's primary source) "
            "before adding it to configs/models.yaml's active_models list."
        )
    return cls
