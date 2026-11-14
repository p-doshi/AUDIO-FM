"""X-ARES encoder wrapper for AudioMAE.

Reuses audio_comp's already-tested AudioMAEEncoder adapter (loading,
resampling, fbank preprocessing) but skips the adapter's mean-pool and
returns the raw (batch, 513, 768) `forward_encoder_no_mask()` output --
cls token + 512 patch tokens, a genuine per-frame sequence, same
category as AST/hubert/mert, not a single-frame workaround like
CLAP/PANNs/Bird-MAE.
"""
from __future__ import annotations

import torch

from audio_comp.models import get_model_class
from audio_comp.models.audiomae import _compute_fbank
from xares_eval.encoders._util import probe_output_dim_and_hop, stack_ragged, to_numpy_mono


class AudioMAEXaresEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._adapter = get_model_class("audiomae")(device="cuda" if torch.cuda.is_available() else "cpu")
        self._adapter.load()
        self.sampling_rate = self._adapter.info.expected_sample_rate
        self.output_dim, self.hop_size_in_ms = probe_output_dim_and_hop(self._forward_one_np, self.sampling_rate)

    def _forward_one_np(self, waveform_np):
        fbank = _compute_fbank(torch.from_numpy(waveform_np).unsqueeze(0), self.sampling_rate)
        batch = fbank.unsqueeze(0).to(self._adapter.device)  # (1, 1, 1024, 128)
        with torch.no_grad():
            contextual_emb = self._adapter._model.forward_encoder_no_mask(batch)
        return contextual_emb[0].cpu()

    def forward(self, audio: torch.Tensor) -> torch.Tensor:
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        sequences = []
        for row in audio:
            waveform_np = to_numpy_mono(row)
            sequences.append(self._forward_one_np(waveform_np))
        return stack_ragged(sequences)


if __name__ == "__main__":
    from xares.audio_encoder_checker import check_audio_encoder

    encoder = AudioMAEXaresEncoder()
    assert check_audio_encoder(encoder)
