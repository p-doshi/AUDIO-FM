"""X-ARES encoder wrapper for Audio-JEPA (ltuncay/Audio-JEPA substitute).
Reuses the adapter's loaded ViT encoder directly — its raw forward output
(128 patches x 768) is already unpooled; audio_comp.models.audio_jepa
mean-pools over patches *after* this for the RDM pipeline."""
from __future__ import annotations

import numpy as np
import torch

from audio_comp.models import get_model_class
from audio_comp.models.audio_jepa import CLIP_LENGTH_S, _compute_mel_spec
from xares_eval.encoders._util import stack_ragged, to_numpy_mono


class AudioJepaXaresEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._adapter = get_model_class("audio_jepa")(device="cuda" if torch.cuda.is_available() else "cpu")
        self._adapter.load()
        self.sampling_rate = self._adapter.info.expected_sample_rate
        # Fixed by construction (see model docstring): 10s input -> 8 temporal
        # patches/s x 16 mel patches, 128 patches total per 10s clip, 768-dim.
        self.output_dim = 768
        self.hop_size_in_ms = 1000 / 8  # 8 temporal positions/s (1.6 effective per config.json's own note)

    def _forward_one_np(self, waveform_np: np.ndarray) -> torch.Tensor:
        min_samples = self.sampling_rate * CLIP_LENGTH_S
        if len(waveform_np) < min_samples:
            waveform_np = np.pad(waveform_np, (0, min_samples - len(waveform_np)))
        wav_tensor = torch.from_numpy(waveform_np).float().unsqueeze(0)
        spec = _compute_mel_spec(wav_tensor).unsqueeze(0).to(self._adapter.device)  # (1, 1, T, F)
        with torch.no_grad():
            out = self._adapter._model(spec)  # (1, 128, 768)
        return out[0].cpu()

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

    encoder = AudioJepaXaresEncoder()
    assert check_audio_encoder(encoder)
