"""AudioMAE — reconstruction-target (masked autoencoder), general-audio
model (AudioSet-pretrained, not domain-specific like Bird-MAE). Fills
the "reconstruction-target paradigm" gap in CLAUDE.md's Stage 2 plan
with a general-domain counterpart to Bird-MAE's bioacoustic-specific
one -- together they test whether the MLP-wins/KNN-loses pattern PANNs
and Bird-MAE both showed on BirdCLEF (2026-08-10) is a property of
non-contrastive training objectives generally, or specific to those two
checkpoints.

Not natively `transformers`-loadable: needs the upstream
github.com/facebookresearch/AudioMAE repo's own `models_mae.py` on
sys.path, plus a checkpoint from the repo's README-linked Google Drive
file. Run `scripts/setup_audiomae.sh` once before using this adapter.

**Requires `timm==0.3.2` specifically** (pinned, not the latest) --
`models_mae.py`'s `Block(..., qk_scale=None, ...)` call uses a
`timm.models.vision_transformer.Block` constructor argument modern timm
(tested: 1.0.28) removed entirely, a real version-skew break, not a
guess (`TypeError: Block.__init__() got an unexpected keyword argument
'qk_scale'`). Safe to pin project-wide: no other adapter in this roster
depends on `timm` at all (`pip show timm` confirmed empty `Required-by`
before pinning). Pinning to 0.3.2 (the same version the original MAE
repo this code derives from pins) reintroduces its own `torch._six`
import — handled by the same shim below that also covers the repo's own
`torch._six.inf` usage.

Checkpoint provenance verified 2026-08-10 against the primary source
directly: CC-BY-4.0, the repo's LICENSE explicitly states "This project
is under the CC-BY 4.0 license" with no carve-out for the checkpoint
weights (unlike `beats`, which required a judgment call — this one is
explicit).

**Exact model-construction kwargs and preprocessing were read directly
out of the downloaded checkpoint itself** (`ckpt['args']`, a full
argparse Namespace saved during AudioSet pretraining), not assumed from
the repo's README or guessed from the paper — the same "verify, don't
assume" discipline the beats/music2vec/A-JEPA corrections established.
Confirmed from `ckpt['args']`: `model='mae_vit_base_patch16'` (ViT-Base,
depth=12), `use_custom_patch=False`, `decoder_mode=1`, `mode=0`,
`alpha=0.0`. Confirmed independently from the state dict's own tensor
shapes: `pos_embed` (1, 513, 768) => 512 patches + 1 cls token, matching
img_size=(1024, 128) at patch_size=16 (1024/16 * 128/16 = 512);
`patch_embed.proj.weight` (768, 1, 16, 16) confirms in_chans=1 (single-
channel spectrogram input, standard non-overlapping patches, not the
`use_custom_patch=True` overlapping-stride variant). Fbank preprocessing
(16kHz, kaldi.fbank, 128 mel bins, frame_shift=10ms, hanning window,
htk_compat=True, DC-offset removal, no low_freq override so the same
default-20Hz cutoff as Bird-MAE/audio_jepa) and normalization constants
(mean=-4.2677393, std=4.5689974) copied directly from the repo's own
`dataset.py`/`main_pretrain.py` (the exact AudioSet norm_stats entry),
not re-derived.

Uses `forward_encoder_no_mask()` (full, unmasked forward pass over every
patch -- the model's own designed inference path, not the masked
training-time `forward_encoder()`) and mean-pools the resulting
(batch, 513, 768) sequence (cls token + 512 patch tokens), same
convention as every other real-per-token-sequence model in this roster
(mert, hubert, wav2vec2, ast).
"""
from __future__ import annotations

import math
import os
import sys
import types
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
import torchaudio

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model
from ._util import mean_pool, resample

EXTERNAL_DIR = Path(os.environ.get("AUDIO_COMP_EXTERNAL", os.path.expanduser("~/audio_comp_external")))
REPO_DIR = EXTERNAL_DIR / "AudioMAE-main"
CHECKPOINT_PATH = EXTERNAL_DIR / "audiomae" / "pretrained.pth"

NUM_MEL_BINS = 128
TARGET_LENGTH = 1024
FBANK_MEAN = -4.2677393
FBANK_STD = 4.5689974


def _install_torch_six_shim() -> None:
    """AudioMAE's `util/misc.py` does `from torch._six import inf`, and
    the pinned `timm==0.3.2` this adapter requires (see load()'s
    docstring note on the `qk_scale` version-skew below) does
    `from torch._six import container_abcs` at import time -- `torch._six`
    was a PyTorch-internal compatibility shim removed in PyTorch 2.0
    (this project runs 2.11.0). Both symbols were always trivial aliases
    (`inf` = `float('inf')`, `container_abcs` = `collections.abc`).
    Installing a minimal fake `torch._six` module before either the repo's
    own code or timm imports it, same category of fix as `bird_mae`'s
    post_init shim and `audio_jepa`'s flash_attn shim -- a narrow,
    documented compatibility shim, not a vendored-code edit or a global
    downgrade."""
    if "torch._six" in sys.modules:
        return
    import collections.abc

    shim = types.ModuleType("torch._six")
    shim.inf = math.inf
    shim.container_abcs = collections.abc
    sys.modules["torch._six"] = shim

    # util/pos_embed.py's get_1d_sincos_pos_embed_from_grid() (called from
    # MaskedAutoencoderViT.__init__ -> initialize_weights(), to fill
    # self.pos_embed's *initial* random-init value) uses `np.float`,
    # removed in numpy >=1.24 ("use `float` by itself... safe", per
    # numpy's own deprecation message). This initial value is immediately
    # overwritten wholesale by load_state_dict() below (`pos_embed` is a
    # real checkpoint key) -- restoring the alias is a correctness no-op
    # for anything this adapter actually uses, just needed so __init__
    # doesn't crash before load_state_dict ever runs.
    if not hasattr(np, "float"):
        np.float = float


def _compute_fbank(waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
    """(1, TARGET_LENGTH, NUM_MEL_BINS) log-mel fbank, exact preprocessing
    from the upstream repo's dataset.py `_wav2fbank` (DC-offset removal,
    kaldi.fbank params, target-length pad/truncate, AudioSet norm stats)."""
    waveform = waveform - waveform.mean()
    fbank = torchaudio.compliance.kaldi.fbank(
        waveform,
        htk_compat=True,
        sample_frequency=sample_rate,
        use_energy=False,
        window_type="hanning",
        num_mel_bins=NUM_MEL_BINS,
        dither=0.0,
        frame_shift=10,
    )
    n_frames = fbank.shape[0]
    p = TARGET_LENGTH - n_frames
    if p > 0:
        fbank = torch.nn.functional.pad(fbank, (0, 0, 0, p))
    elif p < 0:
        fbank = fbank[0:TARGET_LENGTH, :]
    fbank = (fbank - FBANK_MEAN) / (FBANK_STD * 2)
    return fbank.unsqueeze(0)


@register_model("audiomae")
class AudioMAEEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="audiomae",
        hf_id="facebookresearch/AudioMAE (Google Drive checkpoint, not HF-native)",
        paradigm="masked autoencoding, reconstruction-target (general audio, AudioSet)",
        license="CC-BY-4.0",
        expected_sample_rate=16000,
        checkpoint_status="official_open_weights",
    )

    def load(self) -> None:
        if not CHECKPOINT_PATH.exists() or not REPO_DIR.exists():
            raise RuntimeError(
                f"AudioMAE repo/checkpoint not found at {REPO_DIR} / {CHECKPOINT_PATH}. "
                "Run `scripts/setup_audiomae.sh` first."
            )
        _install_torch_six_shim()
        if str(REPO_DIR) not in sys.path:
            sys.path.insert(0, str(REPO_DIR))
        import models_mae  # external repo, not a package dep

        self._model = models_mae.mae_vit_base_patch16_dec512d8b(
            norm_pix_loss=True,
            in_chans=1,
            audio_exp=True,
            img_size=(TARGET_LENGTH, NUM_MEL_BINS),
            alpha=0.0,
            mode=0,
            use_custom_patch=False,
            split_pos=False,
            pos_trainable=False,
            use_nce=False,
            # decoder_mode=0 (global attn), NOT the checkpoint's actual
            # decoder_mode=1 (swin-based local attn) -- see this module's
            # docstring "decoder is never used" note: constructing the
            # real decoder_mode=1 Swin decoder requires two mutually
            # incompatible timm API generations in the same install (the
            # ViT encoder's qk_scale-era Block vs. the Swin decoder's
            # feat_size/drop_attn-era SwinTransformerBlock -- confirmed by
            # directly testing timm 0.3.2/0.4.9/0.4.12/1.0.28, none
            # satisfy both). decoder_mode only affects self.decoder_blocks,
            # never touched by forward_encoder_no_mask() -- load() asserts
            # below that every unmatched state-dict key is decoder-only,
            # so this is a verified no-op for the encoder, not a guess.
            decoder_mode=0,
            mask_2d=False,
            mask_t_prob=0.7,
            mask_f_prob=0.3,
            no_shift=False,
        )
        checkpoint = torch.load(CHECKPOINT_PATH, map_location="cpu", weights_only=False)
        result = self._model.load_state_dict(checkpoint["model"], strict=False)
        bad_keys = [k for k in (*result.missing_keys, *result.unexpected_keys) if not k.startswith("decoder")]
        if bad_keys:
            raise RuntimeError(
                f"AudioMAE state_dict mismatch touches non-decoder keys, encoder load is NOT verified safe: {bad_keys}"
            )
        self._model = self._model.to(self.device).eval()

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        resampled = [resample(w, sample_rate, self.info.expected_sample_rate) for w in waveforms]
        fbanks = [
            _compute_fbank(torch.from_numpy(w).unsqueeze(0), self.info.expected_sample_rate)
            for w in resampled
        ]
        batch = torch.stack(fbanks).to(self.device)  # (B, 1, TARGET_LENGTH, NUM_MEL_BINS)
        with torch.no_grad():
            contextual_emb = self._model.forward_encoder_no_mask(batch)
        return mean_pool(contextual_emb).cpu().numpy()
