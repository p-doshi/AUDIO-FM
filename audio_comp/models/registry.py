"""Model registry: string name -> BaseAudioEncoder subclass."""
from __future__ import annotations

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseAudioEncoder

MODEL_REGISTRY: Dict[str, Type["BaseAudioEncoder"]] = {}


def register_model(name: str):
    def _decorator(cls):
        if name in MODEL_REGISTRY:
            raise ValueError(f"model '{name}' already registered")
        MODEL_REGISTRY[name] = cls
        return cls

    return _decorator


def get_model_class(name: str) -> Type["BaseAudioEncoder"]:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"no model registered as '{name}'. Available: {sorted(MODEL_REGISTRY)}"
        ) from e
