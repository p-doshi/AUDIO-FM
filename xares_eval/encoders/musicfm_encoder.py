"""X-ARES encoder wrapper for MusicFM. Reuses the adapter's get_latent()
call directly — it's already unpooled (audio_comp.models.musicfm mean-pools
*after* get_latent for the RDM pipeline; this wrapper stops one step
earlier)."""
from __future__ import annotations

import numpy as np
import torch

from audio_comp.models import get_model_class
from xares_eval.encoders._util import probe_output_dim_and_hop, stack_ragged, to_numpy_mono


class MusicFMXaresEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._adapter = get_model_class("musicfm")(device="cuda" if torch.cuda.is_available() else "cpu")
        self._adapter.load()
        self.sampling_rate = self._adapter.info.expected_sample_rate
        self.output_dim, self.hop_size_in_ms = probe_output_dim_and_hop(self._forward_one_np, self.sampling_rate)

    def _forward_one_np(self, waveform_np: np.ndarray) -> torch.Tensor:
        wav = torch.from_numpy(waveform_np).unsqueeze(0).to(self._adapter.device)
        with torch.no_grad():
            emb = self._adapter._model.get_latent(wav, layer_ix=self._adapter.layer_ix)
        return emb[0].cpu()

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

    encoder = MusicFMXaresEncoder()
    assert check_audio_encoder(encoder)
