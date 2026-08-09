"""Contract tests for model adapters.

Full load+embed smoke tests hit the network (HF checkpoint downloads,
several GB for the larger models) so they're opt-in via
RUN_MODEL_SMOKE_TESTS=1. The always-on tests just check every registered
adapter follows the BaseAudioEncoder contract, and that the deliberately
deferred models (beats — see its module docstring) fail loudly with
NotImplementedError rather than silently producing garbage.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

from audio_comp.models import MODEL_REGISTRY
from audio_comp.models.base import BaseAudioEncoder

DEFERRED_MODELS = {"beats"}


def test_registry_contains_expected_models():
    expected = {"clap", "mert", "hubert", "wav2vec2", "music2vec", "musicfm", "audio_jepa", "beats"}
    assert expected <= set(MODEL_REGISTRY.keys())


@pytest.mark.parametrize("name", sorted(MODEL_REGISTRY.keys()))
def test_adapter_subclasses_base(name):
    cls = MODEL_REGISTRY[name]
    assert issubclass(cls, BaseAudioEncoder)
    assert cls.info.name == name


@pytest.mark.parametrize("name", sorted(DEFERRED_MODELS))
def test_deferred_models_fail_loudly(name):
    encoder = MODEL_REGISTRY[name](device="cpu")
    with pytest.raises(NotImplementedError):
        encoder.load()


@pytest.mark.skipif(
    os.environ.get("RUN_MODEL_SMOKE_TESTS") != "1",
    reason="downloads real HF checkpoints; set RUN_MODEL_SMOKE_TESTS=1 to run",
)
@pytest.mark.parametrize("name", sorted(set(MODEL_REGISTRY.keys()) - DEFERRED_MODELS))
def test_adapter_embeds_dummy_waveform(name):
    encoder = MODEL_REGISTRY[name](device="cpu")
    encoder.load()
    dummy = np.random.randn(16000).astype(np.float32)  # 1s @ 16kHz; adapters resample as needed
    embedding = encoder.embed(dummy, sample_rate=16000)
    assert embedding.ndim == 1
    assert embedding.shape[0] > 0
    assert np.isfinite(embedding).all()
