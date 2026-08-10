"""X-ARES encoder wrapper for Bird-MAE.

Same situation as CLAP/PANNs (see clap_encoder.py's docstring): despite
its HF output field being named `last_hidden_state`, BirdMAEModel.forward()
already mean-pools over the patch/time axis internally
(`config.global_pool == "mean"`) and returns a single pooled 768-d vector
per clip -- there's no per-frame sequence to expose. Wrapped the same way:
the pooled embedding becomes a single "frame" spanning the whole clip.
`hop_size_in_ms` is nominal here, not load-bearing for a genuinely
single-frame-per-clip output.

Bird-MAE's own feature extractor internally pads/truncates every clip to
a fixed 512-frame (~5s) window regardless of input length (see
audio_comp.models.bird_mae's docstring), so unlike PANNs there's no
separate short-clip floor to enforce here.
"""
from __future__ import annotations

import numpy as np
import torch

from audio_comp.models import get_model_class
from xares_eval.encoders._util import stack_ragged, to_numpy_mono


class BirdMAEXaresEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._adapter = get_model_class("bird_mae")(device="cuda" if torch.cuda.is_available() else "cpu")
        self._adapter.load()
        self.sampling_rate = self._adapter.info.expected_sample_rate
        dummy = np.zeros(self.sampling_rate * 4, dtype=np.float32)
        with torch.no_grad():
            seq = self._forward_one_np(dummy)
        self.output_dim = seq.shape[-1]
        self.hop_size_in_ms = 4000  # one frame spans the whole clip, see module docstring

    def _forward_one_np(self, waveform_np: np.ndarray) -> torch.Tensor:
        features = self._adapter._extractor(
            np.stack([waveform_np]), return_tensors="pt"
        ).to(self._adapter.device)
        with torch.no_grad():
            out = self._adapter._model(input_values=features)
        return out.last_hidden_state[0].unsqueeze(0).cpu()  # (1, output_dim) -- a single frame

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

    encoder = BirdMAEXaresEncoder()
    assert check_audio_encoder(encoder)
