"""Importing this package registers every adapter module with MODEL_REGISTRY."""
from . import audio_jepa, beats, bird_mae, clap, hubert, mert, music2vec, musicfm, panns_cnn14, wav2vec2  # noqa: F401
from .registry import MODEL_REGISTRY, get_model_class

__all__ = ["MODEL_REGISTRY", "get_model_class"]
