"""X-ARES encoder wrapper for Whisper. Distinct pattern from the
wav2vec2-family encoders: WhisperFeatureExtractor always pads/truncates
to a fixed 30s chunk (no padding=True needed, unlike Wav2Vec2FeatureExtractor),
uses "input_features" not "input_values", and only the encoder half of the
full encoder-decoder model is ever called -- matches
audio_comp/models/whisper.py's own policy (never .generate()/the decoder).
"""
from __future__ import annotations

import torch

from audio_comp.models import get_model_class
from xares_eval.encoders._util import probe_output_dim_and_hop, stack_ragged, to_numpy_mono


class WhisperXaresEncoder(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self._adapter = get_model_class("whisper")(device="cuda" if torch.cuda.is_available() else "cpu")
        self._adapter.load()
        self.sampling_rate = self._adapter.info.expected_sample_rate
        self.output_dim, self.hop_size_in_ms = probe_output_dim_and_hop(self._forward_one_np, self.sampling_rate)

    def _forward_one_np(self, waveform_np):
        inputs = self._adapter._extractor([waveform_np], sampling_rate=self.sampling_rate, return_tensors="pt")
        input_features = inputs["input_features"].to(self._adapter.device)
        with torch.no_grad():
            out = self._adapter._model.encoder(input_features)
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

    encoder = WhisperXaresEncoder()
    assert check_audio_encoder(encoder)
