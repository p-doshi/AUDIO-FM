"""Per-layer embedding extraction for the brain-RSA layer-stage-correspondence
test (the paper's actual headline claim: middle layers best predict primary
auditory cortex, deep layers best predict non-primary cortex — v1's
final-layer-only pass in `run_rsa.py` couldn't test this at all).

Deliberately standalone from `audio_comp/models/` rather than adding
`output_hidden_states=True` support to the main adapter interface — every
adapter's `embed()` contract is "one pooled vector per clip" project-wide
(used by the main RSA/CKA/TwoNN pipeline, X-ARES, Stage 5 LoRA fine-tuning),
and this detour only needs per-layer vectors for a handful of models, once.
Mirrors each adapter's own load()/preprocessing exactly (same HF classes,
same feature extractors, same expected sample rate) so the per-layer numbers
are directly comparable to `run_rsa.py`'s final-layer numbers — just adds
output_hidden_states=True and mean-pools every layer instead of only the
last one.

`mert` and `clap` added 2026-08-13 to disambiguate a confound flagged by a
peer session (relayed via cross-session message, verified against the
already-computed `results_per_layer_nh2015.csv` before acting on it — see
journal.md): the original 5-model roster (wav2vec2/hubert/wavlm all
masked+speech, whisper ASR-supervised+speech, ast supervised+general) varies
training objective and training domain together for `ast`, so "ast is the
one model that doesn't differentiate Primary from Lateral cortex by depth"
could be an objective effect OR a domain effect (ast is the only model never
trained on speech) — RSA alone can't distinguish them, same structure as the
music2vec/PANNs confounds already on record in the main pipeline. `mert`
(masked-modeling, music domain) fills the previously-empty
masked-modeling x non-speech-domain cell of that 2x2. `clap` (contrastive,
domain-general) adds a second discriminative-objective, non-speech-domain
data point alongside ast, testing whether "discriminative" or "domain
mismatch" is the better predictor of ast's outlier pattern. `musicfm` (also
masked+music) was suggested too but needs a custom per-layer forward loop
(non-`transformers` architecture, `get_latent(layer_ix=...)` one layer at a
time) — deferred as a follow-up, not done in this pass, to avoid stalling
the cheaper mert/clap disambiguation.

Both new models need resampling (mert: 24kHz, clap: 48kHz; stimuli are
16kHz) and clap's hidden states are Swin-stage 4D feature maps (channels x H
x W), not transformer sequences — spatial-mean-pooled instead of the
seq-mean-pooling the other models use.

Usage:
    python extract_activations_per_layer.py --models wav2vec2 hubert wavlm whisper ast mert clap
"""
from __future__ import annotations

import argparse
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import (
    ASTFeatureExtractor,
    ASTModel,
    ClapModel,
    ClapProcessor,
    HubertModel,
    Wav2Vec2FeatureExtractor,
    Wav2Vec2Model,
    WavLMModel,
    WhisperFeatureExtractor,
    WhisperModel,
)
from transformers import AutoModel

STIM_DIR = Path(__file__).resolve().parent / "auditory_brain_dnn" / "data" / "stimuli" / "165_natural_sounds_16kHz"
OUT_DIR = Path(__file__).resolve().parent / "activations_per_layer"

# (hf_id, feature_extractor_cls, model_cls, expected_sample_rate)
MODEL_SPECS = {
    "wav2vec2": ("facebook/wav2vec2-large-lv60", Wav2Vec2FeatureExtractor, Wav2Vec2Model, 16000),
    "hubert": ("facebook/hubert-large-ll60k", Wav2Vec2FeatureExtractor, HubertModel, 16000),
    "wavlm": ("microsoft/wavlm-base-plus", Wav2Vec2FeatureExtractor, WavLMModel, 16000),
    "whisper": ("openai/whisper-base", WhisperFeatureExtractor, WhisperModel, 16000),
    "ast": ("MIT/ast-finetuned-audioset-10-10-0.4593", ASTFeatureExtractor, ASTModel, 16000),
    "mert": ("m-a-p/MERT-v1-330M", Wav2Vec2FeatureExtractor, AutoModel, 24000),
    "clap": ("laion/larger_clap_general", ClapProcessor, ClapModel, 48000),
}

# Stimuli are all 16kHz; models needing a different rate get resampled per-clip.
NATIVE_SR = 16000


def get_hidden_states(model_name: str, model, extractor, waveform: np.ndarray, sr: int, device: str) -> list[np.ndarray]:
    """Returns a list of (D,) mean-pooled vectors, one per layer (including the
    embedding-output 'layer 0'), for a single clip."""
    if model_name == "whisper":
        inputs = extractor(waveform, sampling_rate=sr, return_tensors="pt")
        input_features = inputs["input_features"].to(device)
        with torch.no_grad():
            enc_out = model.encoder(input_features, output_hidden_states=True)
        hidden_states = enc_out.hidden_states
        return [h.mean(dim=1).squeeze(0).cpu().numpy() for h in hidden_states]
    elif model_name == "clap":
        inputs = extractor(audio=[waveform], sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model.audio_model(**inputs, output_hidden_states=True)
        # Swin-stage feature maps: (1, C, H, W) -> spatial-mean-pool to (C,)
        return [h.mean(dim=(2, 3)).squeeze(0).cpu().numpy() for h in out.hidden_states]
    elif model_name == "mert":
        inputs = extractor(waveform, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        return [h.mean(dim=1).squeeze(0).cpu().numpy() for h in out.hidden_states]
    else:
        inputs = extractor(waveform, sampling_rate=sr, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True)
        hidden_states = out.hidden_states

    return [h.mean(dim=1).squeeze(0).cpu().numpy() for h in hidden_states]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True, choices=list(MODEL_SPECS.keys()))
    args = parser.parse_args()

    wav_paths = sorted(STIM_DIR.glob("*.wav"))
    assert len(wav_paths) == 165, f"expected 165 stimuli, found {len(wav_paths)}"
    stim_ids = [p.stem for p in wav_paths]

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        out_path = OUT_DIR / f"{model_name}.npz"
        if out_path.exists():
            print(f"[{model_name}] already extracted at {out_path}, skipping")
            continue

        hf_id, extractor_cls, model_cls, expected_sr = MODEL_SPECS[model_name]
        trust_remote_code = model_name == "mert"
        print(f"[{model_name}] loading {hf_id} on {device}...")
        extractor = extractor_cls.from_pretrained(hf_id, trust_remote_code=trust_remote_code) \
            if trust_remote_code else extractor_cls.from_pretrained(hf_id)
        model = (model_cls.from_pretrained(hf_id, trust_remote_code=trust_remote_code) if trust_remote_code
                  else model_cls.from_pretrained(hf_id)).to(device).eval()

        per_layer_embeddings: list[list[np.ndarray]] | None = None  # [layer][clip] -> (D,)
        for p in wav_paths:
            waveform, sr = sf.read(p, dtype="float32", always_2d=False)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            assert sr == NATIVE_SR, f"{p} has sr={sr}, expected {NATIVE_SR}"
            if expected_sr != sr:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=expected_sr)
                sr = expected_sr

            layer_vecs = get_hidden_states(model_name, model, extractor, waveform, sr, device)
            if per_layer_embeddings is None:
                per_layer_embeddings = [[] for _ in layer_vecs]
            for i, v in enumerate(layer_vecs):
                per_layer_embeddings[i].append(v)

        n_layers = len(per_layer_embeddings)
        save_dict = {"stim_ids": np.array(stim_ids), "n_layers": n_layers}
        for i, layer_list in enumerate(per_layer_embeddings):
            save_dict[f"layer_{i}"] = np.stack(layer_list)

        np.savez(out_path, **save_dict)
        print(f"[{model_name}] wrote {n_layers} layers x {len(stim_ids)} clips to {out_path}")

        del model
        if device == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
