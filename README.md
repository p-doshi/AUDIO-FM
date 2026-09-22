# ECHO: Embedding Convergence and Hidden Organization in Audio Models

This repository contains the codebase for **ECHO: Embedding Convergence and Hidden Organization in Audio Models**, a study of representational
geometry across independently pretrained audio foundation models, and whether a
model's frozen representation geometry predicts how much it stands to gain from
adaptation before any fine-tuning is run.

This is the code-only release: the paper draft, journal/lab-notebook files, and
result artifacts are intentionally not included in this repository or its history.

## Models

19 independently pretrained audio models, spanning four training paradigms. Every
checkpoint ID below is read directly from this repo's own model adapters
(`audio_comp/models/*.py`), not retyped from memory.

| Model | HF / source ID | Training paradigm |
|---|---|---|
| [AST](https://huggingface.co/MIT/ast-finetuned-audioset-10-10-0.4593) | `MIT/ast-finetuned-audioset-10-10-0.4593` | Supervised / discriminative |
| [PANNs (CNN14)](https://github.com/qiuqiangkong/audioset_tagging_cnn) | `qiuqiangkong/audioset_tagging_cnn` (Zenodo checkpoint) | Supervised / discriminative |
| [Whisper](https://huggingface.co/openai/whisper-base) | `openai/whisper-base` | Supervised / discriminative |
| [CLAP](https://huggingface.co/laion/larger_clap_general) | `laion/larger_clap_general` | Cross-modal contrastive |
| [AudioMAE](https://github.com/facebookresearch/AudioMAE) | `facebookresearch/AudioMAE` (Google Drive checkpoint) | Masked modeling / reconstruction |
| [Bird-MAE](https://huggingface.co/DBD-research-group/Bird-MAE-Base) | `DBD-research-group/Bird-MAE-Base` | Masked modeling / reconstruction |
| [EnCodecMAE](https://github.com/habla-liaa/encodecmae) | `lpepino/encodecmae-large-st` (model name `ec-ec-large_st`) | Masked modeling / reconstruction |
| [HuBERT](https://huggingface.co/facebook/hubert-large-ll60k) | `facebook/hubert-large-ll60k` | Masked modeling / reconstruction |
| [wav2vec 2.0](https://huggingface.co/facebook/wav2vec2-large-lv60) | `facebook/wav2vec2-large-lv60` | Masked modeling / reconstruction |
| [wav2vec 2.0 Conformer](https://huggingface.co/facebook/wav2vec2-conformer-rel-pos-large) | `facebook/wav2vec2-conformer-rel-pos-large` | Masked modeling / reconstruction |
| [WavLM](https://huggingface.co/microsoft/wavlm-base-plus) | `microsoft/wavlm-base-plus` | Masked modeling / reconstruction |
| [UniSpeech-SAT](https://huggingface.co/microsoft/unispeech-sat-base) | `microsoft/unispeech-sat-base` | Masked modeling / reconstruction |
| [SEW](https://huggingface.co/asapp/sew-tiny-100k) | `asapp/sew-tiny-100k` | Masked modeling / reconstruction |
| [MMS](https://huggingface.co/facebook/mms-300m) | `facebook/mms-300m` | Masked modeling / reconstruction |
| [MERT](https://huggingface.co/m-a-p/MERT-v1-330M) | `m-a-p/MERT-v1-330M` | Masked modeling / reconstruction |
| [MusicFM](https://github.com/minzwon/musicfm) | `minzwon/MusicFM` | Masked modeling / reconstruction (BEST-RQ) |
| [Audio-JEPA](https://huggingface.co/ltuncay/Audio-JEPA) | `ltuncay/Audio-JEPA` | Joint-embedding / self-distillation |
| [data2vec-audio](https://huggingface.co/facebook/data2vec-audio-base) | `facebook/data2vec-audio-base` | Joint-embedding / self-distillation |
| [music2vec](https://huggingface.co/m-a-p/music2vec-v1) | `m-a-p/music2vec-v1` | Joint-embedding / self-distillation (data2vec-family, **not** JEPA — see correction below) |

**Correction, kept visible rather than silently fixed:** `music2vec` was originally
labeled JEPA-family in early project notes; it is actually data2vec-family (no
separate predictor network, the architectural line that defines JEPA — see
`audio_comp/models/music2vec.py`'s docstring). `Audio-JEPA` (`ltuncay/Audio-JEPA`) is
an explicitly-labeled substitute for the original A-JEPA paper's checkpoint, which was
never publicly released.

Fourteen models receive both LoRA and ALLoRA adaptation; four (PANNs, AudioMAE,
MusicFM, Audio-JEPA) are adapted only via top-$K$ layer unfreezing, since they either
have no attention layer or embed bespoke, non-differentiable preprocessing that
precludes standard LoRA module injection; EnCodecMAE participates in frozen-probe
geometry only. See `audio_comp/pipelines/allora_model_configs.py` and
`audio_comp/pipelines/topk_unfreeze_configs.py` for the exact target-module naming
per architecture.

## Datasets

| Domain | Source | Link |
|---|---|---|
| Speech | LibriSpeech ASR | [huggingface.co/datasets/openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) |
| Music | FMA-small | [github.com/mdeff/fma](https://github.com/mdeff/fma) |
| City noise | UrbanSound8K | [huggingface.co/datasets/danavery/urbansound8K](https://huggingface.co/datasets/danavery/urbansound8K) (unofficial HF mirror) |
| Bird sounds | BirdCLEF | [huggingface.co/datasets/mteb/birdclef25-mini](https://huggingface.co/datasets/mteb/birdclef25-mini) |
| Machine noise | MIMII (6dB tier) | [zenodo.org/records/3384388](https://zenodo.org/records/3384388) |
| Vessel | confidential hydrophone recordings | not publicly available; used under a restrictive data-sharing agreement |

`data/probe_set_manifest.csv` is the only probe-set artifact tracked in git; raw
audio and per-domain fine-tuning clip pools are not included in this repository.

## Codebase structure

```
audio_comp/
  models/              one file per model adapter, registered by name (audio_comp/models/base.py)
  data/                dataset source loaders + probe-set builder
  geometry/            RDM / RSA / CKA / TwoNN intrinsic dimension / uniformity
  pipelines/           extraction, fine-tuning (LoRA/ALLoRA/top-K), and geometry
                       computation entry points -- see "What runs what" below
configs/
  models.yaml          which registered models are active in a given run
  categories.yaml      which dataset source backs each probe-set category
scripts/
  appendix_reproduction/   scripts to regenerate the paper's appendix results
                            (accuracy table, RSA heatmaps, partial-correlation table)
  finding6a_self_rsa_*.py  per-domain self-RSA computation (frozen vs. adapted RDM)
  physics/                 separate side-investigation, code only
  slurm/                   cluster job submission scripts
data/
  probe_set_manifest.csv, *_manifest.csv   per-domain clip manifests (paths, labels, folds)
xares_eval/            X-ARES-based downstream evaluation harness
brain_rsa/              separate side-track (neural-data RSA comparison)
```

## What runs what

The pipeline has four stages, run in this order:

1. **Probe-set construction.** `python -m audio_comp.data.build_probe_set` writes
   `data/probe_set_manifest.csv` from the dataset sources under
   `audio_comp/data/sources/`.

2. **Frozen embedding extraction.** One Slurm job per model
   (`scripts/slurm/extract_embeddings.sbatch` -> `audio_comp/pipelines/extract_embeddings.py`),
   writing one `.npz` (embeddings + clip IDs) per model. `audio_comp/pipelines/finetune_data_geometry.py`
   does the same for each domain's own fine-tuning clip pool (a different, larger set
   than the general probe set).

3. **Adaptation.** `audio_comp/pipelines/generic_lora_trainer.py` (LoRA/ALLoRA) and
   `audio_comp/pipelines/*_finetune_*.py` / `topk_unfreeze_configs.py` (top-$K$) fine-tune
   each eligible model on each domain, writing per-(model, domain, condition, seed)
   accuracy and a full per-clip adapted representational dissimilarity matrix (RDM)
   checkpoint (used for self-RSA in step 4).

4. **Geometry analysis.** `audio_comp/pipelines/compare_models.py` (pretrained RSA/CKA),
   `audio_comp/pipelines/rsa_cka_finetuned.py` (adapted RSA/CKA), `audio_comp/geometry/`
   (uniformity, TwoNN), and `scripts/finding6a_self_rsa_*.py` (self-RSA, frozen-vs-adapted
   per model) produce the geometry metrics used throughout the analysis.

`scripts/appendix_reproduction/` contains the scripts used to regenerate the paper's
appendix results (full accuracy table, pretrained/finetuned RSA heatmaps, the
self-RSA partial-correlation table) from the outputs of steps 1-4 above.

## Setup

Neither `audio_comp` nor `xares_eval` are `pip install -e .`'d — both are
`PYTHONPATH`-relative packages; export `PYTHONPATH="$HOME/audio_comp"` from the repo
root before running anything under `audio_comp.pipelines.*` or `xares_eval.*`.
