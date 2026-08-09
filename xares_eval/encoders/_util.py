"""Shared helpers for X-ARES encoder wrappers.

These reuse the already-tested audio_comp.models adapters' loaded
underlying HF models, but expose UNPOOLED frame-level hidden states
(X-ARES's expected `forward(audio) -> (batch, time, output_dim)`
interface) instead of audio_comp's pooled per-clip embeddings — the two
frameworks want different things from the same underlying model. Not
optimized for throughput (loops per-sample within a batch); correctness
over speed for a first working integration.
"""
from __future__ import annotations

import numpy as np
import torch


def to_numpy_mono(row: torch.Tensor) -> np.ndarray:
    return row.detach().cpu().numpy().astype(np.float32)


def stack_ragged(sequences: list[torch.Tensor]) -> torch.Tensor:
    """Zero-pad a list of (T_i, D) tensors to (batch, max_T, D)."""
    max_len = max(seq.shape[0] for seq in sequences)
    dim = sequences[0].shape[1]
    out = torch.zeros(len(sequences), max_len, dim, dtype=sequences[0].dtype)
    for i, seq in enumerate(sequences):
        out[i, : seq.shape[0]] = seq
    return out


def probe_output_dim_and_hop(forward_one, sampling_rate: int, probe_seconds: float = 4.0) -> tuple[int, float]:
    """Run one dummy forward pass to determine output_dim and hop_size_in_ms
    empirically, rather than hardcoding per-model constants that could
    silently go stale if a checkpoint's architecture changes."""
    dummy = np.zeros(int(sampling_rate * probe_seconds), dtype=np.float32)
    with torch.no_grad():
        seq = forward_one(dummy)  # (T, D)
    output_dim = seq.shape[-1]
    hop_size_in_ms = (probe_seconds * 1000) / seq.shape[0]
    return output_dim, hop_size_in_ms
