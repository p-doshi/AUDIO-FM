"""Extract final-layer (pooled) embeddings for our foundation models on the
Tuckute/Feather/McDermott 165-natural-sound stimulus set.

v1 scope: final layer only, via each model's existing `embed()` pooled
output (same call `audio_comp/pipelines/extract_embeddings.py` uses for the
main probe set). Per-layer extraction is a follow-up step once this quick
pass gives a sanity-checkable result.

Usage:
    python extract_activations.py --models wav2vec2 hubert wavlm whisper ast
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

import sys
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
from audio_comp.models import get_model_class  # noqa: E402

STIM_DIR = Path(__file__).resolve().parent / "auditory_brain_dnn" / "data" / "stimuli" / "165_natural_sounds_16kHz"
OUT_DIR = Path(__file__).resolve().parent / "activations"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    args = parser.parse_args()

    wav_paths = sorted(STIM_DIR.glob("*.wav"))
    assert len(wav_paths) == 165, f"expected 165 stimuli, found {len(wav_paths)}"
    stim_ids = [p.stem for p in wav_paths]  # e.g. 'stim5_alarm_clock'

    device = "cuda" if torch.cuda.is_available() else "cpu"
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for model_name in args.models:
        out_path = OUT_DIR / f"{model_name}.npz"
        if out_path.exists():
            print(f"[{model_name}] already extracted at {out_path}, skipping")
            continue

        print(f"[{model_name}] loading on {device}...")
        encoder = get_model_class(model_name)(device=device)
        encoder.load()

        embeddings = []
        for p in wav_paths:
            waveform, sr = sf.read(p, dtype="float32", always_2d=False)
            if waveform.ndim > 1:
                waveform = waveform.mean(axis=1)
            embeddings.append(encoder.embed(waveform, sr))
        embeddings = np.stack(embeddings)

        np.savez(out_path, stim_ids=np.array(stim_ids), embeddings=embeddings)
        print(f"[{model_name}] wrote {embeddings.shape} to {out_path}")


if __name__ == "__main__":
    main()
