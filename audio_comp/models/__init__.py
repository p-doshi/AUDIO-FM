"""Importing this package registers every adapter module with MODEL_REGISTRY."""
from . import (  # noqa: F401
    ast,
    audio_jepa,
    audiomae,
    beats,
    bird_mae,
    clap,
    data2vec_audio,
    hubert,
    mert,
    mms,
    music2vec,
    musicfm,
    panns_cnn14,
    sew,
    unispeech_sat,
    wav2vec2,
    wav2vec2_conformer,
    wavlm,
    whisper,
)
from .registry import MODEL_REGISTRY, get_model_class

__all__ = ["MODEL_REGISTRY", "get_model_class"]
