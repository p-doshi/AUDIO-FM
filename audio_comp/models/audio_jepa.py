"""Audio-JEPA (ltuncay/Audio-JEPA) — JEPA-family, substitute for the original
paper's A-JEPA, which has no public checkpoint anywhere.

NOT the original A-JEPA (Fei, Fan, Huang, arXiv 2311.15830) — this is an
independently-built, similarly-named model (Tuncay et al., ICME 2025,
github.com/LudovicTuncay/Audio-JEPA). Always label it as a substitute in any
write-up that references it.

Wiring notes: `JEPA.ckpt` is a PyTorch Lightning checkpoint whose
LightningModule takes Hydra-instantiated `encoder`/`predictor`/`criterion`
submodules that are excluded from the saved hyperparameters, so
`load_from_checkpoint(ckpt_path)` alone doesn't work. Rather than
reconstructing the full Hydra/Lightning training config, this loads just
the ViT encoder directly: the HF repo ships an author-provided
`inference_example.py` (alongside `config.json`, which corroborates every
architecture constant below) that does exactly this — instantiate
`VisionTransformer` standalone, load only the `encoder.*` keys from the
checkpoint's state dict, and discard predictor + target_encoder (per
config.json: "Only the encoder is used downstream. Predictor + target
encoder are discarded at inference."). This module ports that approach.

`vision_transformer.py`'s `Block` unconditionally imports
`flash_attn.modules.mha.MHA` (a compiled CUDA extension) even when
`use_flash_attn=False` is passed — the import happens regardless of the
flag. `_install_flash_attn_shim` below replaces it with a torch-native
`scaled_dot_product_attention`-based module (matching the checkpoint's
`qkv`/`proj` parameter names for `load_state_dict(strict=True)`), ported
from the HF repo's own inference example — no flash-attn build needed, and
SDPA gets CUDA-optimized kernels automatically when run on GPU.

Compute/undertraining confound to account for when this is compared:
this checkpoint is trained on meaningfully less compute than the other
active teachers — 100k steps (~14h on 4 V100s, 5,338h of AudioSet) vs.
wav2vec2/data2vec's 400k steps on larger batches. The paper's own results
show it substantially underperforming both baselines on several
linear-probe tasks (e.g. Speech Commands V1: 0.152 vs. data2vec's 0.927).
If this model comes back RSA-isolated from the other five, that's
confounded between "JEPA-paradigm geometry is genuinely different" (what
this project wants to test) and "this checkpoint is comparatively
undertrained" — RSA alone can't distinguish them. Run
`audio_comp/pipelines/inspect_geometry.py` on it the same way music2vec's
isolation was disambiguated (see the 2026-08-09 journal entry). The paper
states its objective favors embedding cohesion over linear separability
(strong kNN, weak linear probe on the same tasks) — predicting a *low*
intrinsic dimension and/or unusually tight within-category clustering
relative to the other five, the opposite direction from music2vec (which
had the *highest* ID of the six active models).

One-time setup: `bash scripts/setup_audio_jepa.sh` (clones the source repo;
the checkpoint itself is fetched via huggingface_hub inside `load()`, same
as any other HF-hosted model).
"""
from __future__ import annotations

import os
import sys
import types
from importlib.machinery import ModuleSpec
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torch.nn as nn
import torchaudio

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import resample

EXTERNAL_DIR = Path(os.environ.get("AUDIO_COMP_EXTERNAL", os.path.expanduser("~/audio_comp_external")))
REPO_DIR = EXTERNAL_DIR / "audio-jepa"

# Architecture/preprocessing constants, cross-confirmed by the HF repo's
# inference_example.py and config.json (both first-party, not reverse-engineered).
SAMPLE_RATE = 32_000
CLIP_LENGTH_S = 10
N_MELS = 128
TARGET_TIME_BINS = 256
PATCH_SIZE = (16, 16)
EMBED_DIM = 768
DEPTH = 12
NUM_HEADS = 12
MLP_RATIO = 4.0


def _install_flash_attn_shim() -> None:
    """Replace flash_attn.modules.mha.MHA with a CPU/GPU-portable torch class.

    Matches the checkpoint's parameter naming (qkv, proj) so
    load_state_dict(..., strict=True) succeeds without a flash-attn build.
    """

    class _PortableMultiHeadAttention(nn.Module):
        def __init__(
            self,
            embed_dim: int,
            num_heads: int,
            dropout: float = 0.0,
            qkv_proj_bias: bool = True,
            use_flash_attn: bool = False,
            **_ignore,
        ) -> None:
            super().__init__()
            assert embed_dim % num_heads == 0
            self.num_heads = num_heads
            self.head_dim = embed_dim // num_heads
            self.dropout = dropout
            self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=qkv_proj_bias)
            self.proj = nn.Linear(embed_dim, embed_dim, bias=True)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            b, n, d = x.shape
            qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, self.head_dim)
            q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
            out = torch.nn.functional.scaled_dot_product_attention(
                q, k, v, dropout_p=self.dropout if self.training else 0.0
            )
            return self.proj(out.transpose(1, 2).reshape(b, n, d))

    def _make(name: str) -> types.ModuleType:
        m = types.ModuleType(name)
        m.__spec__ = ModuleSpec(name, loader=None)
        return m

    flash_attn = _make("flash_attn")
    flash_attn.__version__ = "0.0.0-portable-shim"
    modules = _make("flash_attn.modules")
    mha = _make("flash_attn.modules.mha")
    mha.MHA = _PortableMultiHeadAttention
    sys.modules.update(
        {"flash_attn": flash_attn, "flash_attn.modules": modules, "flash_attn.modules.mha": mha}
    )


def _stub_upstream_inits(root: Path) -> None:
    """Pre-empt heavy __init__.py files in the upstream repo that pull in
    hydra/wandb/lightning — only the leaf model-definition files are needed."""
    for cached in list(sys.modules):
        if cached == "src" or cached.startswith("src."):
            del sys.modules[cached]
    for name in (
        "src",
        "src.utils",
        "src.models",
        "src.models.components",
        "src.masks",
        "src.masks.components",
    ):
        m = types.ModuleType(name)
        m.__path__ = [str(root / name.replace(".", "/"))]
        sys.modules[name] = m


def _compute_mel_spec(waveform: torch.Tensor) -> torch.Tensor:
    """Kaldi-fbank mel spectrogram of shape (1, TARGET_TIME_BINS, N_MELS)."""
    hop_length_ms = (CLIP_LENGTH_S * 1000) / TARGET_TIME_BINS
    frame_length_ms = 2.5 * hop_length_ms
    spec = torchaudio.compliance.kaldi.fbank(
        waveform - waveform.mean(),
        sample_frequency=SAMPLE_RATE,
        frame_length=frame_length_ms,
        frame_shift=hop_length_ms,
        num_mel_bins=N_MELS,
        low_freq=20,
        high_freq=SAMPLE_RATE // 2,
        use_log_fbank=True,
        window_type="hanning",
    )
    if spec.shape[0] < TARGET_TIME_BINS:
        spec = torch.cat([spec, torch.zeros(TARGET_TIME_BINS - spec.shape[0], N_MELS)], dim=0)
    elif spec.shape[0] > TARGET_TIME_BINS:
        spec = spec[:TARGET_TIME_BINS]
    return spec.unsqueeze(0)


@register_model("audio_jepa")
class AudioJepaEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="audio_jepa",
        hf_id="ltuncay/Audio-JEPA",
        paradigm="JEPA-family (general audio; substitute for the unavailable original A-JEPA)",
        license="MIT",
        expected_sample_rate=SAMPLE_RATE,
    )

    def load(self) -> None:
        if not (REPO_DIR / "src" / "models" / "components" / "vision_transformer.py").exists():
            raise RuntimeError(
                f"Audio-JEPA source not found at {REPO_DIR}. Run scripts/setup_audio_jepa.sh first."
            )
        _install_flash_attn_shim()
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))
        _stub_upstream_inits(REPO_DIR)

        from src.models.components.vision_transformer import VisionTransformer

        encoder = VisionTransformer(
            input_size=(TARGET_TIME_BINS, N_MELS),
            patch_size=PATCH_SIZE,
            in_chans=1,
            embed_dim=EMBED_DIM,
            depth=DEPTH,
            num_heads=NUM_HEADS,
            mlp_ratio=MLP_RATIO,
            use_flash_attn=False,
        )

        from huggingface_hub import hf_hub_download

        ckpt_path = hf_hub_download(self.info.hf_id, "JEPA.ckpt")
        state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        raw_sd = state.get("state_dict", state)
        # Only `encoder.*` — predictor and target_encoder are discarded at
        # inference per the model's own config.json.
        encoder_sd = {
            k[len("encoder.") :]: v
            for k, v in raw_sd.items()
            if k.startswith("encoder.") and not k.startswith("encoder_")
        }
        missing, unexpected = encoder.load_state_dict(encoder_sd, strict=True)
        if missing or unexpected:
            raise RuntimeError(
                f"Audio-JEPA encoder state_dict mismatch: {len(missing)} missing, "
                f"{len(unexpected)} unexpected keys"
            )
        self._model = encoder.to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        min_samples = self.info.expected_sample_rate * CLIP_LENGTH_S
        specs = []
        for waveform in waveforms:
            resampled = resample(waveform, sample_rate, self.info.expected_sample_rate)
            if len(resampled) < min_samples:
                # kaldi.fbank's fixed ~98ms analysis window needs the input
                # at least that long; some probe-set clips (e.g. brief
                # UrbanSound8K events) are shorter. Pad to the model's
                # designed 10s input rather than relying on fbank to cope
                # with arbitrary short waveforms.
                resampled = np.pad(resampled, (0, min_samples - len(resampled)))
            wav_tensor = torch.from_numpy(resampled).float().unsqueeze(0)  # (1, n_samples)
            specs.append(_compute_mel_spec(wav_tensor))  # (1, T, F)
        batch = torch.stack(specs).to(self.device)  # (B, 1, T, F)
        with torch.no_grad():
            embeddings = self._model(batch)  # (B, 128 patches, 768)
        return embeddings.mean(dim=1).cpu().numpy()  # mean-pool over patches
