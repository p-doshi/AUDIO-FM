"""X-ARES encoder wrapper for CLAP.

CLAP is fundamentally a clip-level contrastive model, not a frame-level
one: its audio tower (HTSAT, a Swin-Transformer-based encoder) produces a
last_hidden_state that's a 2D time-frequency spatial feature map
(measured shape: (batch, 1024, 2, 32)), not a 1D temporal sequence — there
is no natural, non-arbitrary way to flatten that into X-ARES's expected
(batch, time, output_dim) frame sequence without an unjustified choice of
which spatial axis is "time". Rather than force an arbitrary reshape,
this wrapper uses CLAP's own pooled contrastive embedding
(get_audio_features(), the representation CLAP is actually designed and
trained to produce) as a single "frame" spanning the whole clip: shape
(1, output_dim). This is honest about what CLAP actually is, and is
sufficient for the clip-level classification tasks this project uses
X-ARES for (X-ARES's MLP/kNN protocols pool/aggregate frame embeddings
for clip-level tasks regardless).
"""
from __future__ import annotations

import numpy as np
import torch

from audio_comp.models import get_model_class
from xares_eval.encoders._util import stack_ragged, to_numpy_mono


class ClapXaresEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._adapter = get_model_class("clap")(device="cuda" if torch.cuda.is_available() else "cpu")
        self._adapter.load()
        self.sampling_rate = self._adapter.info.expected_sample_rate
        dummy = np.zeros(self.sampling_rate * 4, dtype=np.float32)
        with torch.no_grad():
            seq = self._forward_one_np(dummy)
        self.output_dim = seq.shape[-1]
        self.hop_size_in_ms = 4000  # one frame spans the whole clip, see module docstring

    def _forward_one_np(self, waveform_np: np.ndarray) -> torch.Tensor:
        inputs = self._adapter._processor(
            audio=[waveform_np], sampling_rate=self.sampling_rate, return_tensors="pt"
        )
        inputs = {k: v.to(self._adapter.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._adapter._model.get_audio_features(**inputs)
        return out.pooler_output[0].unsqueeze(0).cpu()  # (1, output_dim) -- a single frame

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

    encoder = ClapXaresEncoder()
    assert check_audio_encoder(encoder)
