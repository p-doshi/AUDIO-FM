"""Master model-report table -- one row per active model, meant as the
single reference table for any future chart/grouping (by domain, by
objective, by architecture, by input representation, by scale). Every
column is either read directly from each adapter's own ModelInfo /
registry metadata, or verified by direct instantiation (parameter count,
embedding dimension) -- the same discipline as count_parameters.py,
never copied from a secondary source without checking.

Columns that are NOT derivable from code (domain category, architecture
family in human terms, input representation, training-objective bucket)
were hand-verified against each adapter's own feature-extractor code
and, where that wasn't enough, each model's primary paper/repo directly
(2026-08-14) -- see MANUAL_METADATA below for the specific reasoning per
model, not asserted from memory.

Usage:
    python -m audio_comp.pipelines.model_report --out results/model_report.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]

# Domain category: what the model's OWN training data actually is, not
# what task it's being evaluated on. "general" = trained on broad,
# non-domain-specific audio (AudioSet-scale or broader), not matched to
# any single one of this project's 5 probe categories.
DOMAIN_CATEGORY = {
    "hubert": "speech", "wav2vec2": "speech", "wavlm": "speech",
    "data2vec_audio": "speech", "mms": "speech", "unispeech_sat": "speech",
    "sew": "speech", "wav2vec2_conformer": "speech", "whisper": "speech",
    "mert": "music", "music2vec": "music", "musicfm": "music",
    "bird_mae": "bird_sounds",
    "clap": "general", "ast": "general", "audiomae": "general",
    "panns_cnn14": "general", "audio_jepa": "general",
}

# Training objective bucket -- matches Stage 4's discriminative/
# self-supervised framing (CLAUDE.md), broken out further for this table.
TRAINING_OBJECTIVE = {
    "hubert": "masked_modeling", "wav2vec2": "masked_modeling", "wavlm": "masked_modeling",
    "unispeech_sat": "masked_modeling+speaker_aware", "sew": "masked_modeling",
    "wav2vec2_conformer": "masked_modeling", "mms": "masked_modeling",
    "mert": "masked_modeling", "musicfm": "masked_modeling_bestrq",
    "data2vec_audio": "self_distillation_data2vec", "music2vec": "self_distillation_data2vec",
    "audio_jepa": "self_distillation_jepa",
    "audiomae": "reconstruction_mae", "bird_mae": "reconstruction_mae",
    "clap": "contrastive", "ast": "supervised_classification",
    "panns_cnn14": "supervised_classification", "whisper": "supervised_asr",
}

# Coarse discriminative-vs-not, as used in scaling_analysis.py.
DISCRIMINATIVE = {"ast", "clap", "panns_cnn14"}


def discriminative_bucket(name: str) -> str:
    if name in DISCRIMINATIVE:
        return "discriminative"
    if name == "whisper":
        return "supervised_asr"
    return "self_supervised"


# Input representation: what the network's core architecture actually
# operates on (not just what the Python API accepts -- e.g. panns_cnn14's
# loader takes raw audio, but the CNN internally computes a log-mel
# spectrogram before its first conv layer, so it's classified as
# spectrogram here). Verified per-model 2026-08-14:
#  - Wav2Vec2FeatureExtractor-based models (hubert/wav2vec2/wavlm/
#    data2vec_audio/mms/unispeech_sat/sew/wav2vec2_conformer/mert/
#    music2vec): raw_waveform -- this feature extractor only normalizes
#    raw audio, no spectrogram conversion, confirmed by reading each
#    adapter's embed_batch().
#  - ast: ASTFeatureExtractor produces a log-mel spectrogram (ViT-style
#    patches over it).
#  - whisper: WhisperFeatureExtractor always produces an 80-channel
#    log-mel spectrogram (well-documented, fixed 30s chunks).
#  - clap: HTSAT (Swin-Transformer) audio branch operates on a 64-dim
#    log-mel spectrogram, confirmed via primary source (LAION-CLAP paper).
#  - musicfm: BEST-RQ scheme, confirmed via primary paper (arXiv
#    2311.03318) -- 80-dim log-mel spectrogram input, not raw waveform.
#  - audio_jepa: explicit torchaudio.compliance.kaldi.fbank in
#    audio_jepa.py -- mel-filterbank spectrogram.
#  - audiomae/bird_mae: both explicit fbank/spectrogram computation in
#    their own adapter code (_compute_fbank / bundled feature extractor),
#    standard ViT-MAE-on-spectrogram-patches design.
#  - panns_cnn14: CNN14's forward() computes log-mel spectrogram
#    internally before the conv stack (confirmed in panns_cnn14.py's own
#    docstring).
INPUT_REPRESENTATION = {
    "hubert": "raw_waveform", "wav2vec2": "raw_waveform", "wavlm": "raw_waveform",
    "data2vec_audio": "raw_waveform", "mms": "raw_waveform", "unispeech_sat": "raw_waveform",
    "sew": "raw_waveform", "wav2vec2_conformer": "raw_waveform", "mert": "raw_waveform",
    "music2vec": "raw_waveform",
    "ast": "spectrogram", "whisper": "spectrogram", "clap": "spectrogram",
    "musicfm": "spectrogram", "audio_jepa": "spectrogram", "audiomae": "spectrogram",
    "bird_mae": "spectrogram", "panns_cnn14": "spectrogram",
}

# Architecture family, in human terms -- verified against each model's
# primary paper/repo, not guessed from the paradigm string alone.
ARCHITECTURE_TYPE = {
    "hubert": "transformer_encoder_cnn_frontend", "wav2vec2": "transformer_encoder_cnn_frontend",
    "wavlm": "transformer_encoder_cnn_frontend", "data2vec_audio": "transformer_encoder_cnn_frontend",
    "mms": "transformer_encoder_cnn_frontend", "unispeech_sat": "transformer_encoder_cnn_frontend",
    "sew": "transformer_encoder_cnn_frontend_squeezed", "mert": "transformer_encoder_cnn_frontend",
    "music2vec": "transformer_encoder_cnn_frontend",
    "wav2vec2_conformer": "conformer_encoder_cnn_frontend",
    "musicfm": "conformer_encoder",
    "whisper": "transformer_encoder_decoder_cnn_frontend",
    "ast": "vision_transformer",
    "audiomae": "vision_transformer_mae", "bird_mae": "vision_transformer_mae",
    "clap": "swin_transformer_htsat",
    "panns_cnn14": "pure_cnn",
    "audio_jepa": "vision_transformer_jepa_predictor",
}


def load_param_counts(path: Path) -> dict[str, int]:
    counts = {}
    with open(path) as f:
        for row in csv.DictReader(f):
            if row["n_params"]:
                counts[row["model"]] = int(row["n_params"])
    return counts


def get_embed_dims(active_models: list[str]) -> dict[str, int]:
    from audio_comp.models import get_model_class

    dims = {}
    for name in active_models:
        print(f"[{name}] checking embed dim...", flush=True)
        try:
            adapter = get_model_class(name)(device="cpu")
            adapter.load()
            dummy = np.zeros(int(adapter.info.expected_sample_rate * 2.0), dtype=np.float32)
            out = adapter.embed_batch([dummy], adapter.info.expected_sample_rate)
            dims[name] = int(out.shape[-1])
            print(f"[{name}] embed_dim={dims[name]}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED: {e}", flush=True)
        finally:
            if "adapter" in dir():
                del adapter
    return dims


def main(out_csv: str) -> None:
    with open(REPO_ROOT / "configs" / "models.yaml") as f:
        active_models = sorted(yaml.safe_load(f)["active_models"])

    params = load_param_counts(REPO_ROOT / "results" / "model_parameter_counts.csv")
    embed_dims = get_embed_dims(active_models)

    from audio_comp.models import get_model_class

    rows = []
    for name in active_models:
        info = get_model_class(name).info
        rows.append({
            "name": name,
            "category": DOMAIN_CATEGORY.get(name, ""),
            "training_objective": TRAINING_OBJECTIVE.get(name, ""),
            "discriminative_bucket": discriminative_bucket(name),
            "input_representation": INPUT_REPRESENTATION.get(name, ""),
            "architecture_type": ARCHITECTURE_TYPE.get(name, ""),
            "n_params": params.get(name, ""),
            "embed_dim": embed_dims.get(name, ""),
            "native_sample_rate_hz": info.expected_sample_rate,
            "hf_id": info.hf_id,
            "license": info.license,
            "checkpoint_status": info.checkpoint_status,
        })

    # birdnet: not in the registry (isolated TF pipeline), but part of
    # the RSA roster and the earlier category-distribution discussion --
    # included here with what's verified, blank for what isn't (no
    # reliable primary-source parameter count found 2026-08-14, see
    # journal.md; do not guess one in).
    rows.append({
        "name": "birdnet",
        "category": "bird_sounds",
        "training_objective": "supervised_classification",
        "discriminative_bucket": "discriminative",
        "input_representation": "spectrogram",
        "architecture_type": "efficientnet_cnn",
        "n_params": "",
        "embed_dim": "",
        "native_sample_rate_hz": 48000,
        "hf_id": "kahst/BirdNET-Analyzer (Zenodo v2.4, TFLite -- not HF-native)",
        "license": "CC BY-NC-SA 4.0 (weights); MIT (code)",
        "checkpoint_status": "official_open_weights",
    })

    out_path = Path(out_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(REPO_ROOT / "results" / "model_report.csv"))
    args = parser.parse_args()
    main(args.out)
