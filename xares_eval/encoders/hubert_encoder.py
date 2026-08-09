"""X-ARES encoder wrapper for HuBERT.

Reuses audio_comp's already-tested HubertEncoder adapter for loading and
resampling, but returns the unpooled last_hidden_state (X-ARES wants
frame-level embeddings; audio_comp's adapter mean-pools for the RDM
pipeline). See xares_eval/encoders/_util.py for why this file exists
separately rather than modifying the adapter.
"""
from __future__ import annotations

import torch

from audio_comp.models import get_model_class
from xares_eval.encoders._util import probe_output_dim_and_hop, stack_ragged, to_numpy_mono


class HubertXaresEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._adapter = get_model_class("hubert")(device="cuda" if torch.cuda.is_available() else "cpu")
        self._adapter.load()
        self.sampling_rate = self._adapter.info.expected_sample_rate
        self.output_dim, self.hop_size_in_ms = probe_output_dim_and_hop(self._forward_one_np, self.sampling_rate)

    def _forward_one_np(self, waveform_np):
        inputs = self._adapter._extractor(
            [waveform_np], sampling_rate=self.sampling_rate, return_tensors="pt", padding=True
        )
        inputs = {k: v.to(self._adapter.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._adapter._model(**inputs)
        return out.last_hidden_state[0].cpu()

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

    encoder = HubertXaresEncoder()
    assert check_audio_encoder(encoder)
