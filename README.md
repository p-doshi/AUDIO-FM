# ECHO: Embedding Convergence and Hidden Organization in Audio Models

This repository contains the code, data manifests, and results for **ECHO**, a study
of representational geometry across 19 independently pretrained audio foundation
models, and whether a model's frozen representation geometry predicts how much it
stands to gain from adaptation before any fine-tuning is run. The full paper draft is
[`final_iclr.tex`](final_iclr.tex).

## Abstract

> Pretrained models trained once on large-scale datasets are now routinely repurposed
> for entirely new tasks, but there is no reliable way to predict, in advance, how much
> a given model will benefit from adaptation without first fine-tuning it. This is a
> real cost in domains where labeled data is scarce, forcing researchers to spend weeks
> and compute on trial-and-error. We propose ECHO: a concrete, falsifiable geometric
> signature of adaptation headroom, together with a precise map of the tasks where that
> signature holds and where it does not. We show that a single, measurable property of
> a model's internal representation — independent of its training data, architecture,
> or task performance — predicts how much a model stands to gain from adapting to a new
> task, before any adaptation is performed. We study numerous independently trained
> audio models spanning speech, music, environmental sound, bioacoustics, and machine
> acoustics. We measure how each model internally arranges sound relative to every
> other model, and find that agreement between models is real but only partially
> explained by what they were trained on — no single factor accounts for it fully. On
> tasks with real learnable signal, models with more clustered representations gain
> roughly 3.7x more from adaptation than already-spread models. Self-RSA further shows
> that this relationship is reflected in actual geometric reorganization during
> fine-tuning across three structurally different adaptation mechanisms. Domain-matched
> pretraining is therefore not sufficient for model selection: general-purpose models
> with no domain-specific training outperform domain-specialist models on their own
> home tasks by up to 7%.

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
geometry only. See `final_iclr.tex` §3.2 / Appendix A.2 (Table 2) for the exact
target-module naming per architecture.

## Datasets

| Domain | Source | Link | Clips |
|---|---|---|---|
| Speech | LibriSpeech ASR | [huggingface.co/datasets/openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) | 10,026 |
| Music | FMA-small | [github.com/mdeff/fma](https://github.com/mdeff/fma) | 8,000 |
| City noise | UrbanSound8K | [huggingface.co/datasets/danavery/urbansound8K](https://huggingface.co/datasets/danavery/urbansound8K) (unofficial HF mirror) | 8,732 |
| Bird sounds | BirdCLEF | [huggingface.co/datasets/mteb/birdclef25-mini](https://huggingface.co/datasets/mteb/birdclef25-mini) | 6,335 |
| Machine noise | MIMII (6dB tier) | [zenodo.org/records/3384388](https://zenodo.org/records/3384388) | 2,400 (seeded subsample of 18,019 raw files) |
| Vessel (confidential) | JASCO AMAR hydrophone recordings, Canada's East Coast | not publicly linkable — class labels are restricted under a data-sharing agreement (see Ethics Statement, `final_iclr.tex`) | 2,463 |

Exact per-domain clip counts, sample rates, and class-balance notes are in Table 1 of
`final_iclr.tex`. `data/probe_set_manifest.csv` is the only probe-set artifact tracked
in git (an older, superseded 8-category/2,000-clip-per-category pilot manifest — see
`CLAUDE.md`'s frozen/provisional banner for why it no longer matches the paper's
current 6-domain setup); raw audio and per-domain fine-tuning clip pools live on
cluster scratch storage, not in this repo.

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
  appendix_reproduction/   scripts + README to regenerate the paper's appendix
                            (accuracy table, RSA heatmaps, partial-correlation table)
  finding6a_self_rsa_*.py  per-domain self-RSA computation (frozen vs. adapted RDM)
  physics/                 separate side-investigation, out of scope for the ECHO paper
                            itself -- see PHYSICS_INVESTIGATION_README.md
  slurm/                   cluster job submission scripts
data/
  probe_set_manifest.csv, *_manifest.csv   per-domain clip manifests (paths, labels, folds)
results/               every accuracy/RSA/CKA/self-RSA result reported in the paper
                        (see "Results and figures" below)
figures_appendix/      the appendix's combined RSA heatmap figure
xares_eval/            X-ARES-based downstream evaluation harness (Findings 1-2)
brain_rsa/             separate side-track (neural-data RSA comparison), not part of
                        the ECHO paper's own claims
journal.md             running lab notebook (append-only; the authoritative record of
                        what was tried, what failed, and why)
CLAUDE.md              full project scope, standing conventions, and finding-by-finding
                        history (much more detailed than this README)
```

## What runs what

The pipeline has four stages, run in this order:

1. **Probe-set construction.** `python -m audio_comp.data.build_probe_set` writes
   `data/probe_set_manifest.csv` from the dataset sources under
   `audio_comp/data/sources/`.

2. **Frozen embedding extraction.** One Slurm job per model
   (`scripts/slurm/extract_embeddings.sbatch` -> `audio_comp/pipelines/extract_embeddings.py`),
   writing one `.npz` (embeddings + clip IDs) per model to `$SCRATCH`.
   `audio_comp/pipelines/finetune_data_geometry.py` does the same for each domain's own
   fine-tuning clip pool (a different, larger set than the general probe set).

3. **Adaptation.** `audio_comp/pipelines/generic_lora_trainer.py` (LoRA/ALLoRA) and
   `audio_comp/pipelines/*_finetune_*.py` / `topk_unfreeze_configs.py` (top-$K$) fine-tune
   each eligible model on each domain, writing per-(model, domain, condition, seed)
   accuracy to `results/finetune_*.csv` / `results/stage5_*_mimii.csv` and a full
   per-clip adapted representational dissimilarity matrix (RDM) checkpoint (used for
   self-RSA in step 4) — see the Reproducibility Statement in `final_iclr.tex` for why
   these RDM checkpoints (~512 GB total) aren't distributed with this repo.

4. **Geometry analysis.** `audio_comp/pipelines/compare_models.py` (pretrained RSA/CKA),
   `audio_comp/pipelines/rsa_cka_finetuned.py` (adapted RSA/CKA), `audio_comp/geometry/`
   (uniformity, TwoNN), and `scripts/finding6a_self_rsa_*.py` (self-RSA, frozen-vs-adapted
   per model) produce every geometry number reported in the paper.

**To regenerate the paper's appendix specifically** (full accuracy table, pretrained/
finetuned RSA heatmaps, the 18-cell self-RSA partial-correlation table in Finding 3),
see [`scripts/appendix_reproduction/README.md`](scripts/appendix_reproduction/README.md) —
it documents the exact script order, a real checkpoint-precedence bug that was found
and fixed during this work, and what data is and isn't redistributable.

## Results and figures

**Main paper figures** (`final_iclr.tex` §3, §4): the methodology diagram
(`ICLR_m.png`, Figure 1) and the taxonomy-RSA alignment forest plot (`F1.png`, Figure 2,
converted from `results/figures/mantel_forest_bh_bootstrap.svg`). Note: `F1.png` color-codes
domain labels by an earlier "regime" (Structured/Intermediate/Quasi-stationary) framework
that is not otherwise referenced in the current paper text (the framework was walked back
earlier in the project's history) — kept as-is per explicit direction, not an oversight.

**Appendix figures** (`figures_appendix/`):
- `heatmap_combined_2x2.png` — the paper's Appendix Figure (pretrained 19x19,
  finetuned LoRA/ALLoRA 14x14, finetuned top-$K$ 4x4 RSA matrices, one panel each)
- `heatmap_pretrained_19x19.png`, `heatmap_finetuned_14x14_lora.png`,
  `heatmap_finetuned_14x14_allora.png`, `heatmap_finetuned_4x4_topk.png` — the same
  four panels as standalone images

**Key results files** (`results/`, non-exhaustive — see `CLAUDE.md` and `journal.md`
for the full history of every intermediate/superseded file):
- `appendix_big_accuracy_table.csv` — every model x domain x condition accuracy
  (mean +/- std across seeds/folds), backing Appendix Table 3
- `appendix_pretrained_rsa_19x19_domain_avg.csv`,
  `appendix_finetuned_rsa_{14x14_lora,14x14_allora,4x4_topk}_avg.csv` — the matrices
  behind the appendix heatmaps
- `finding6_partial_corr_18cell.csv` — the self-RSA-vs-gain bivariate and partial
  (controlling for uniformity) correlations for all 6 domains x 3 adaptation
  conditions, backing Finding 3's Table 4
- `finetune_{fma_genre,urbansound8k,birdclef,librispeech_speaker}.csv`,
  `stage5_{frozen,lora,allora,topk}_mimii.csv`, `vessel_all_experiments.csv` — raw
  per-(model, condition, seed[, fold]) accuracy, one file set per domain
- `finding6a_self_rsa_*.csv` — per-domain self-RSA (frozen-vs-adapted RDM correlation)
- `rsa_finetuned_matrix_*.csv`, `vessel_rsa_matrix_*.csv`,
  `cka_finetuned_matrix_*.csv` — per-(domain, condition, seed) cross-model RSA/CKA
  matrices restricted to the LoRA/ALLoRA/top-$K$-eligible model subsets
- `model_arch_taxonomy.csv`, `model_roster.csv` — model metadata (paradigm,
  architecture, breadth-hypothesis grouping) used throughout the analysis scripts

## Setup

See `CLAUDE.md` for the full project scope and standing conventions, and
[`scripts/appendix_reproduction/README.md`](scripts/appendix_reproduction/README.md)
for environment notes specific to the appendix-reproduction scripts (thread-limiting,
the `scipy-stack` cluster module for `matplotlib`). Neither `audio_comp` nor
`xares_eval` are `pip install -e .`'d — both are `PYTHONPATH`-relative packages; export
`PYTHONPATH="$HOME/audio_comp"` from the repo root before running anything under
`audio_comp.pipelines.*` or `xares_eval.*`.
