"""Importing this package registers every dataset source with DATASET_REGISTRY."""
from . import sources  # noqa: F401
from .registry import DATASET_REGISTRY, get_dataset_class

__all__ = ["DATASET_REGISTRY", "get_dataset_class"]
