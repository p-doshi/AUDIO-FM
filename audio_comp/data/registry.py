"""Dataset source registry: string name -> BaseDatasetSource subclass."""
from __future__ import annotations

from typing import Dict, Type, TYPE_CHECKING

if TYPE_CHECKING:
    from .base import BaseDatasetSource

DATASET_REGISTRY: Dict[str, Type["BaseDatasetSource"]] = {}


def register_dataset(name: str):
    def _decorator(cls):
        if name in DATASET_REGISTRY:
            raise ValueError(f"dataset source '{name}' already registered")
        DATASET_REGISTRY[name] = cls
        return cls

    return _decorator


def get_dataset_class(name: str) -> Type["BaseDatasetSource"]:
    try:
        return DATASET_REGISTRY[name]
    except KeyError as e:
        raise KeyError(
            f"no dataset source registered as '{name}'. Available: {sorted(DATASET_REGISTRY)}"
        ) from e
