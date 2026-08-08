"""Shared helpers used by multiple model adapters."""
from __future__ import annotations

import librosa
import numpy as np
import torch


def resample(waveform: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return waveform
    return librosa.resample(waveform, orig_sr=orig_sr, target_sr=target_sr)


def mean_pool(hidden_states: torch.Tensor) -> torch.Tensor:
    """Mean-pool a (batch, time, dim) tensor over the time axis -> (batch, dim)."""
    return hidden_states.mean(dim=1)
