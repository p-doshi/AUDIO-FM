"""X-ARES encoder wrapper for PANNs Cnn14.

Like CLAP (see clap_encoder.py's docstring for the fuller rationale),
Cnn14 is architecturally clip-level, not frame-level: its forward pass
pools out the time axis entirely inside the model (`torch.mean` over the
frequency axis, then a max+mean pool over the remaining time axis) before
returning a single 2048-d 'embedding' vector — there is no per-frame
sequence to expose. Wrapped the same way as CLAP: the pooled embedding
becomes a single "frame" spanning the whole clip. `hop_size_in_ms` is
nominal here (matches the 4s dummy probe duration) — it isn't load-
bearing for a genuinely single-frame-per-clip output, same as CLAP's.
"""
from __future__ import annotations

import numpy as np
import torch

from audio_comp.models import get_model_class
from xares_eval.encoders._util import stack_ragged, to_numpy_mono


class PANNsCnn14XaresEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._adapter = get_model_class("panns_cnn14")(device="cuda" if torch.cuda.is_available() else "cpu")
        self._adapter.load()
        self.sampling_rate = self._adapter.info.expected_sample_rate
        dummy = np.zeros(self.sampling_rate * 4, dtype=np.float32)
        with torch.no_grad():
            seq = self._forward_one_np(dummy)
        self.output_dim = seq.shape[-1]
        self.hop_size_in_ms = 4000  # one frame spans the whole clip, see module docstring

    def _forward_one_np(self, waveform_np: np.ndarray) -> torch.Tensor:
        # Same short-clip floor as audio_comp.models.panns_cnn14.embed_batch
        # (Cnn14's pooling stack collapses to a zero-size dimension below
        # ~400ms) -- X-ARES's own dataloader doesn't know about this
        # model-specific minimum, so it must be enforced here too.
        min_samples = int(self._adapter.MIN_SAFE_SAMPLES_S * self.sampling_rate)
        if len(waveform_np) < min_samples:
            waveform_np = np.pad(waveform_np, (0, min_samples - len(waveform_np)))
        wav = torch.from_numpy(waveform_np).unsqueeze(0).to(self._adapter.device)
        with torch.no_grad():
            out = self._adapter._model(wav)
        return out["embedding"][0].unsqueeze(0).cpu()  # (1, output_dim) -- a single frame

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

    encoder = PANNsCnn14XaresEncoder()
    assert check_audio_encoder(encoder)
