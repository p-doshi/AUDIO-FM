"""Audio-JEPA (ltuncay/Audio-JEPA) — JEPA-family, substitute for the original
paper's A-JEPA, which has no public checkpoint anywhere.

NOT the original A-JEPA (Fei, Fan, Huang, arXiv 2311.15830) — this is an
independently-built, similarly-named model (Tuncay et al., ICME 2025,
github.com/LudovicTuncay/Audio-JEPA). Always label it as a substitute in any
write-up that references it.

Deferred from the kickoff run: `JEPA.ckpt` is a PyTorch Lightning checkpoint
whose LightningModule (`JEPAModule` in that repo's src/models/jepa_module.py)
takes Hydra-instantiated `encoder`/`predictor`/`criterion` submodules as
constructor arguments that are explicitly excluded from the saved
hyperparameters (`save_hyperparameters(..., ignore=[...])`). That means
`JEPAModule.load_from_checkpoint(ckpt_path)` does NOT work with just the
checkpoint path — the ViT-Base encoder has to be reconstructed via that
repo's Hydra configs (`configs/model/*.yaml`), matched to whichever config
produced the released checkpoint, before `load_state_dict` can restore
weights. Per the model's HF card: ViT-Base encoder (12 layers, 768-dim, 12
heads), 10s/32kHz mono input, (256, 128) log-mel spectrogram -> (128, 768)
output embedding. This is real integration work, not a config tweak — wire
it up as a fast-follow once the pipeline is validated on the HF-native
models.

Compute/undertraining confound to account for once this is wired up and
compared: this checkpoint is trained on meaningfully less compute than the
other active teachers — 100k steps (~14h on 4 V100s, 5,338h of AudioSet)
vs. wav2vec2/data2vec's 400k steps on larger batches. The paper's own
results show it substantially underperforming both baselines on several
linear-probe tasks (e.g. Speech Commands V1: 0.152 vs. data2vec's 0.927).
If this model comes back RSA-isolated from the other five, that's
confounded between "JEPA-paradigm geometry is genuinely different" (what
this project wants to test) and "this checkpoint is comparatively
undertrained" (unrelated to the paradigm question) — RSA alone can't
distinguish them. Run `audio_comp/pipelines/inspect_geometry.py` on it the
same way music2vec's isolation was disambiguated (see the 2026-08-09
journal entry): specifically check TwoNN intrinsic dimension and
within-category clustering tightness. The paper states its objective
favors embedding cohesion over linear separability (strong kNN, weak
linear probe on the same tasks) — that predicts a *low* intrinsic
dimension and/or unusually tight within-category clustering relative to
the other five, the opposite direction from music2vec (which had the
*highest* ID of the six active models). A low-ID/high-cohesion result
would be real, paper-grounded evidence for paradigm-driven geometry
differences; without that check, an isolated RSA number alone can't
distinguish paradigm from undertraining.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import BaseAudioEncoder, ModelInfo
from .registry import register_model


@register_model("audio_jepa")
class AudioJepaEncoder(BaseAudioEncoder):
    info = ModelInfo(
        name="audio_jepa",
        hf_id="ltuncay/Audio-JEPA",
        paradigm="JEPA-family (general audio; substitute for the unavailable original A-JEPA)",
        license="MIT",
        expected_sample_rate=32000,
    )

    def load(self) -> None:
        raise NotImplementedError(
            "Audio-JEPA's checkpoint requires reconstructing its Hydra-configured "
            "ViT encoder before load_state_dict works — see this module's docstring "
            "for the exact integration path."
        )

    def embed_batch(self, waveforms: Sequence[np.ndarray], sample_rate: int) -> np.ndarray:
        raise NotImplementedError("see load()")
