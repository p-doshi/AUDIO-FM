# ECHO: reproducibility and implementation details

This document gives the full experimental specification behind the paper: datasets and probe-set
composition, splits, model configuration, adaptation hyper-parameters, the accuracy probe, the geometry
metrics and every statistical test. The paper's main text gives the essentials and points here for the rest.

Conventions used throughout:

- **Domain** is one of six audio domains: music, speech, city (urban) sound, bird sound, machine sound and vessel sound.
- **Pool** is the set of clips used for a domain. Every per-domain quantity (geometry, RSA, probe accuracy) is
  computed on that domain's pool.
- **Condition** is one of: frozen, LoRA, ALLoRA, top-*K*.
- **Seeds** are 0–4 for every adapted condition. Frozen extraction is deterministic and has no seed.

---

## 1. Datasets and probe-set composition

### 1.1 Overview

| Domain | Source | Pool (clips) | Classes | Label | Sample rate | Crop | Class skew (max:min) |
|---|---|---:|---:|---|---|---|---|
| Music | FMA-small | 7,997 | 8 | genre | 44.1 kHz | first 10 s | 1:1 |
| Speech | LibriSpeech (train-clean-100 speakers) | 10,026 | 251 | speaker identity | 16 kHz | first 10 s | 1.5:1 |
| City | UrbanSound8K | 8,732 | 10 | sound class | mixed (8–192 kHz) | none (clips ≤ 4 s) | 2.7:1 |
| Bird | BirdCLEF 2025 mini (`mteb/birdclef25-mini`) | 6,335 | 50 | species | 32 kHz | 5 s chunks | 7.2:1 |
| Machine | MIMII, 6 dB tier | 2,400 | 2 | normal / abnormal | 16 kHz | none | 1:1 |
| Vessel | restricted-access underwater recordings | 2,463 | 6 | vessel class | withheld | withheld | withheld |

- Native sample rates are not normalized across domains. Each clip is read at its true rate and resampled to the
  rate each encoder expects (`read_mono` / `read_batch_skip_bad` in `audio_comp/pipelines/generic_lora_trainer.py`).
- Music and speech clips are cropped to their first 10 s (`MAX_CLIP_DURATION_S`). The same crop is used for
  extraction and for adaptation.
- Music has 7,997 clips instead of 8,000 because three FMA-small MP3s cannot be decoded. They are dropped for
  every model.

### 1.2 Per-domain construction and the split used for accuracy and gain

Accuracy and gain use one fixed train/validation/test split per domain. The same split is used for every
model, condition and seed.

| Domain | Split rule | Train | Val | Test |
|---|---|---:|---:|---:|
| Music | FMA-small's native 80/10/10 split (`data/fma_genre_manifest.csv`, column `split`) | 6,397 | 800 | 800 |
| Speech | 70/15/15 split stratified per speaker. Splits are disjoint at the utterance level; all 251 speakers appear in every split (closed-set identification). | 7,018 | 1,503 | 1,505 |
| City | Test = UrbanSound8K fold 1. A fixed 15% of folds 2–10 is held out for validation (seed 0). | 6,681 | 1,178 | 873 |
| Bird | Test = recording-level fold 0. Each species has 20 source recordings, assigned round-robin to 5 folds (4 recordings per species per fold), so no test recording contributes training clips. A fixed 15% of the training pool is held out for validation (seed 0). | 4,111 | 725 | 1,499 |
| Machine | Leave-one-machine-type-out: test = fan, train = pump, slider and valve. A fixed 15% of the training pool is held out for selecting *C* (seed 0). | 1,530 | 270 | 600 |

Per-domain details:

- **Machine pool.** The 2,400 clips are 300 per (machine type, condition) for 4 machine types × {normal,
  abnormal}. They are sampled with a fixed seed (`CLIP_POOL_SEED = 0`), so pool membership does not depend on
  the run seed. All 8 groups have at least 456 clips available.
- **Bird.** Each recording is cut into consecutive 5 s chunks (`xares_eval/birdclef/build_manifest.py`). Clips
  from one recording are near-duplicates, which is why folds are assigned at the recording level.
- **City.** The pool is all 8,732 UrbanSound8K clips.

### 1.3 Vessel domain

The vessel recordings are used under a data-sharing agreement and are not distributed. We release no raw audio,
labels, per-clip outputs or per-seed values for this domain; we report only aggregate statistics.

The analysis code is the same as for the public domains. Vessel experiments run separately on a restricted
server, and only model-level aggregates leave it.

### 1.4 Other probe material in this repository

- `data/probe_set_manifest.csv` is a shared multi-domain probe set built by `audio_comp.data.build_probe_set`
  (seed 42, `configs/categories.yaml`). It is used for exploratory analyses only. All results in the paper use
  the per-domain pools above.
- `additional_results/` holds aggregate tables for analyses outside the paper. Its README describes them.

---

## 2. Models

### 2.1 Checkpoints

The 19 checkpoints and their training-paradigm grouping are listed in `README.md` (section *Models*). Each
model has one adapter in `audio_comp/models/`, registered by name.

### 2.2 Clip embedding

- Each clip gives one fixed-size vector, the mean over time of the encoder's final hidden-state sequence
  (`audio_comp/models/_util.py: mean_pool`).
- Four models use their native clip-level pooling instead: AST, CLAP, PANNs and Bird-MAE.
- MusicFM is read at its intermediate layer 7 (`get_latent(layer_ix=7)`).
- Embeddings are stored as float32. All downstream computation is in float64.

### 2.3 Adaptation eligibility

| Condition | Models |
|---|---|
| LoRA and ALLoRA | AST, Bird-MAE, CLAP, data2vec-audio, HuBERT, MERT, MMS, music2vec, SEW, UniSpeech-SAT, wav2vec 2.0, wav2vec 2.0 Conformer, WavLM, Whisper (14) |
| Top-*K* unfreezing | PANNs (no attention layers), Audio-JEPA, AudioMAE, MusicFM (preprocessing that cannot be wrapped for adapter injection) (4) |
| Frozen only | EnCodecMAE |

Results are always compared within one condition. Conditions are never pooled.

---

## 3. Adaptation

All adaptation runs train a linear classification head on the pooled embedding with cross-entropy, using the
split in §1.2.

| Setting | Value |
|---|---|
| Optimizer | AdamW, learning rate 1e-4 |
| Batch size | 16 (wav2vec 2.0 Conformer: 4, for memory) |
| Epochs | up to 8, early stopping on validation accuracy (patience 3); the best-validation state is kept |
| Seeds | 0, 1, 2, 3, 4 (seed sets `torch`, `numpy` and the training order) |
| LoRA | rank 8, α = 16, dropout 0.05 (`peft`) |
| LoRA targets | attention query and value projections: `q_proj`/`v_proj` (wav2vec2-family, HuBERT, WavLM, UniSpeech-SAT, SEW, MMS, data2vec-audio, MERT, music2vec, AST, Whisper); `linear_q`/`linear_v` (wav2vec 2.0 Conformer); fused `qkv` (Bird-MAE); `audio_model.*.(query\|value)` (CLAP audio tower) |
| ALLoRA | rank 8, η = 8.0, same target modules as LoRA (`audio_comp/pipelines/allora.py`, `allora_model_configs.py`) |
| Top-*K* | the last *K* = 2 blocks plus the final projection or norm are unfrozen. PANNs: `conv_block5–6` + `fc1`. Audio-JEPA and AudioMAE: last 2 transformer blocks + `norm` (the AudioMAE decoder is not used). MusicFM: last 2 of 12 conformer layers (layers 10–11). |

Notes on top-*K* and outputs:

- MusicFM's embedding is read at layer 7, before the unfrozen layers. Its top-*K* adapted embedding therefore
  differs from the frozen one only through the classification objective's effect on layers it cannot reach.
  Read MusicFM top-*K* results with this in mind.
- After training, the adapted encoder embeds the domain's full pool (train, val and test, in loader order). The
  run then stores the clip × clip correlation-distance RDM and the Gram matrix *E Eᵀ* for each
  (domain, condition, model, seed).
- Top-*K* target names are in `audio_comp/pipelines/topk_unfreeze_configs.py`. LoRA targets are in
  `audio_comp/pipelines/lora_model_configs.py`.

---

## 4. Accuracy probe and adaptation gain

Frozen and adapted accuracy use the same probe, applied to features recovered from each encoder's Gram matrix.
Both sides go through an identical pipeline.

**Feature recovery.** These steps are identical for frozen and adapted encoders:

1. Build the Gram matrix *G = E Eᵀ* over the full pool and double-centre it: *H G H*.
2. Take the top *d* eigenpairs (*d* = the encoder's embedding width, at most *n − 1*) and clip negative
   eigenvalues to zero.
3. Whiten each principal direction to roughly unit variance, with a small ridge:
   *X = V · √(λ / (λ + 10⁻³ · mean λ)) · √n*.

This gives the embeddings up to a rotation. The whitening is the rotation-invariant counterpart of per-feature
standardization.

**Probe.**

- L2-regularized logistic regression (scikit-learn, L-BFGS, `max_iter = 3000`).
- *C* is selected on the validation split from {10⁻², 10⁻¹, 1, 10, 10², 10³}. The probe is refit on train with
  the chosen *C* and scored on test.
- The probe is fit separately for each (domain, model, condition, seed).

**Gain** is *A*_adapted − *A*_frozen on the test split, both from this probe.

- Adapted checkpoints that are numerically identical to the frozen encoder are detected and excluded.
- The adaptation condition used for gain is LoRA for the 14 LoRA-eligible models and top-*K* for the other 4.
  ALLoRA gains are computed the same way and reported separately.
- Machine has no validation split in its loader, so a fixed 15% of its training pool (seed 0) is held out only to
  choose *C*.
- **Code:** `scripts/paper/converged_probe_gain.py`; per-seed outputs in `results/probes/converged/`.

---

## 5. Frozen usability probes (Finding 2)

- **Split:** the test split of §1.2. Machine uses a stratified 80/20 split instead.
- **Features:** standardized with a scaler fit on train.
- **kNN:** *k* = 10, Euclidean distance on standardized features.
- **Linear:** logistic regression, *C* = 1, L-BFGS (`max_iter = 500`, `tol = 1e-3`; the accuracies quoted in the
  text). The same model is also run to convergence (`max_iter = 10000`, `tol = 1e-6`), and the converged accuracy
  is used for the kNN − linear gap. Only 1 of 94 fits reaches the 500-iteration cap.
- **kNN-to-linear ratio:** kNN accuracy / linear accuracy. Per domain, the mean ratio of AST and CLAP is compared
  with the mean ratio of the masked-modelling / reconstruction models. Two-sided sign test over the five public
  domains (5/5, *p* = 0.0625). With all four supervised or contrastive models (AST, CLAP, PANNs, Whisper) the
  count is 4/5 (*p* = 0.375); both are in `results/paper_outputs/finding2_per_domain.csv`.
- **Uniformity vs. kNN accuracy and vs. the kNN − linear gap:** Spearman ρ pooled over the 94 (model, domain)
  rows after demeaning every variable within domain. Also reported as a partial correlation controlling for
  log parameter count: the ranks of demeaned uniformity and kNN accuracy are residualized on the rank of log
  parameter count (not demeaned, since it is a model property), and the Spearman correlation of the residuals is
  reported, with the variance inflation factor of the two predictors.
- **Code:** `scripts/paper/finding2_frozen_probes.py` (probes, needs embeddings) and `scripts/paper/finding2.py`
  (statistics, runs from `results/`).
- **Label grain:** five coarse-to-fine label hierarchies:
  - speaker sex → speaker identity
  - machine type → unit
  - bird species → source recording
  - sound class → source file
  - genre → artist

  Fine classes need at least 10 clips. Model rankings under coarse and fine labels are compared with the
  agreement between two random resplits. Whether the change is organized by training paradigm is tested with a
  pre-specified Kruskal–Wallis test. Code and aggregate output are in `scripts/additional_analyses/multigrain_ext_*.py`
  and `additional_results/multigrain/`.

---

## 6. Geometry metrics

All metrics are computed within one domain, on that domain's full pool, from the frozen clip embeddings of §2.2.

| Metric | Definition | Implementation |
|---|---|---|
| RDM | *RDM*ᵢⱼ = 1 − Pearson(*zᵢ*, *zⱼ*) | `audio_comp/geometry/rdm.py` (`scipy.spatial.distance.pdist`, `metric="correlation"`) |
| RSA | Spearman ρ between the upper triangles (*i < j*) of two models' RDMs on the same clips in the same order | `audio_comp/geometry/rsa.py` |
| Linear CKA | ‖*Yᵀ X*‖²_F / (‖*Xᵀ X*‖_F ‖*Yᵀ Y*‖_F) on column-centred embeddings; secondary measure only | `audio_comp/geometry/cka.py` |
| Uniformity | log of the mean over *i ≠ j* of exp(−2 ‖*ẑᵢ* − *ẑⱼ*‖²), with *ẑ* L2-normalized; *t* = 2; all clips in the pool | `audio_comp/geometry/alignment_uniformity.py: uniformity_score` |
| Alignment | mean squared distance between L2-normalized same-label pairs | `alignment_uniformity.py: alignment_score` |
| TwoNN intrinsic dimension | `skdim.id.TwoNN()` on raw embeddings; compared only between models within one domain | `audio_comp/geometry/intrinsic_dim.py` |
| Shared offset | ‖mean of L2-normalized embeddings‖², frozen vs. adapted | `scripts/paper/adaptation_geometry.py` (from stored Gram matrices) |
| Self-RSA | RSA between a model's frozen RDM and its adapted RDM on the same pool | `scripts/self_rsa_and_heatmaps/compute_self_rsa.py` |

- Clips that fail to decode for any model are dropped for all models. Embeddings are realigned by clip path
  before any cross-model comparison.
- Machine clip file names repeat across machine, unit and condition, so machine clips are keyed by their last
  four path components.

---

## 7. Statistical analyses

Every analysis below is implemented in `scripts/paper/` (see its README for the run order); full result tables are
in `APPENDIX.md`.

### Finding 1: cross-domain stability of model similarity

- **Matrices.** Each domain gives a 19 × 19 RSA matrix (§6).
- **Stability.** For each of the 15 domain pairs, stability is the Spearman ρ between the two matrices' upper
  triangles (171 model pairs).
- **Significance.** Each model enters 18 of the 171 pairs, so the pairs are not independent. Significance comes
  from a one-sided permutation test that shuffles model identities (rows and columns together) in one matrix
  and keeps the other fixed. It uses 10,000 permutations with `numpy.random.default_rng(0)`, and
  *p* = (1 + #{null ≥ observed}) / (1 + permutations).
- **Correction.** Benjamini–Hochberg (BH) over the 15 pairs at *q* = 0.05.
- **Robustness.** The test is repeated with each model removed in turn: 19 rosters of 18 models, 5,000
  permutations each, with BH applied separately within each roster.

### Finding 1: taxonomy alignment

- **Groupings.** Two groupings of the 19 models, defined in `audio_comp/pipelines/breadth_hypothesis_check.py`:
  1. Training domain and paradigm (`DOMAIN_PARADIGM_GROUPS`).
  2. Training-data breadth, domain-specific vs. mixed (`BREADTH_GROUPS`).
- **Statistic.** The Mantel statistic is the Pearson correlation between a domain's RSA upper triangle and a
  same-group indicator.
- **Test.** One-sided model-identity permutation test (200,000 permutations, rng 0); 95% CI from 10,000 bootstrap
  resamples of models. BH over the 12 domain × grouping tests. Leave-one-model-out: 10,000 permutations per roster.
- **Code:** `scripts/paper/finding1.py` (both Finding 1 analyses; the vessel matrix is passed with `ECHO_VESSEL_RSA`).

### Finding 3: does frozen geometry predict gain?

**Regression.**

- **Model:** gain ~ difficulty + uniformity + domain match.
- **Rows:** every (model, domain, seed) with a converged-probe gain in the primary condition: 428 rows covering
  18 models (all except EnCodecMAE) and five public domains.
- **Predictors:**
  - *Difficulty* is a domain-level term: the domain's mean frozen probe accuracy over models, chance-normalized
    as (*a* − *c*) / (1 − *c*), where *c* = 1 / number of classes.
  - *Domain match* is 1 if the domain is the model's home domain, else 0. Speech: HuBERT, wav2vec 2.0,
    WavLM, UniSpeech-SAT, SEW, wav2vec 2.0 Conformer, MMS, data2vec-audio, Whisper. Music: MERT, MusicFM,
    music2vec. Bird: Bird-MAE.
- **Uncertainty:**
  - Standard errors are clustered by model.
  - 95% confidence intervals come from 2,000 bootstrap resamples of models.
  - *R²* is reported with a bootstrap CI.
- **Code:** `scripts/paper/finding3_gain_regression.py`.

**Within-domain ranking.**

- Gain and uniformity are averaged over seeds per (model, domain), then demeaned within domain.
- Spearman ρ is computed over the 90 pairs, with a model-bootstrap CI.

**Sensitivity analyses.**

- The regression is refit without machine, where all models are near chance.
- The regression is refit on speech LoRA using only complete seeds.
- Vessel is analysed separately with 18 models.

**Predictor search.**

- The unit is (model, domain), averaged over seeds. Every variable is demeaned within domain.
- Every subset of one to three predictors is drawn from: uniformity, alignment, TwoNN, log parameter count,
  domain match, and a discriminative-objective flag (AST, PANNs, Whisper, CLAP).
- Each subset is scored by leave-one-model-out cross-validated *R²*, where all of a model's domains are held
  out together.
- The best subset is compared with a 2,000-permutation null that permutes model identity, the same
  permutation in every domain, and re-runs the full search each time, so the null includes the subset selection.
- Searches: 2 targets (gain, adapted accuracy) × 3 pools (label-free; plus frozen kNN, linear and converged
  frozen accuracy; plus the post-hoc change in uniformity) = 6.
- **Code:** `scripts/paper/finding3_predictor_search.py`.

**Transferability baselines.** Within domain, Spearman ρ between gain and:

- LogME on frozen embeddings (mean over classes of the per-sample log evidence of a one-vs-rest Bayesian
  linear model).
- kNN accuracy (k = 5, cosine similarity) with up to 100 labelled reference clips per class (at most half of
  each class, seed 0), evaluated on the rest of the domain's pool.
- Estimators are computed by `scripts/paper/transferability_estimators.py`.

Both use two-sided model-identity permutation *p*-values (5,000 draws) and 2,000-model-bootstrap CIs
(`scripts/paper/finding3_transferability.py`; estimator values in `results/probes/transferability_estimators.csv`).

**Negative control.** Adding frozen probe accuracy as a predictor is reported with Bonferroni correction over
the six searches. Because gain = adapted − frozen accuracy, this term mostly measures headroom and regression
to the mean.

### Adaptation and geometry (descriptive)

- **Shared offset.** Frozen vs. adapted, counted over valid adaptation runs (primary condition and ALLoRA).
- **ΔRSA.** For each seed, the mean over models of (mean adapted RSA to the other models − mean frozen RSA to
  the same models); reported as mean and SD over seeds, per domain and method. WavLM is excluded in every domain.
- **Before/after agreement.** Spearman ρ across models between frozen and seed-averaged adapted mean RSA, per
  domain × method cell (10 cells), two-sided permutation *p* (50,000 draws) and BH over the 10 cells.
- **Code:** `scripts/paper/finding3_reorganization.py`.

### Self-RSA links

- **Links tested.** Two links per domain × method cell (18 cells):
  1. Uniformity → self-RSA.
  2. Partial self-RSA → gain, controlling for uniformity (rank residualization). Gain is the converged-probe
     gain of the same condition (§4).
- **Primary analysis.** At model level, with seeds averaged. *p*-values come from model permutation (20,000
  draws); top-*K* cells have four models, so they use the exact 24-ordering permutation (smallest attainable
  two-sided *p* = 0.083). BH is applied over the cells. In bird top-*K* the four models' uniformity and self-RSA
  ranks are perfectly reversed, so the partial correlation is undefined there.
- **Seed-level analysis.** Reported only as a footnote, because seeds share pretrained weights and are not
  independent.

Output: `results/paper_outputs/finding3_self_rsa_links.csv`; inputs in `results/geometry/adaptation/self_rsa_model_level.csv`.

---

### Robustness checks

Code: `scripts/paper/robustness_frozen.py` (frozen embeddings) and `scripts/paper/robustness_analysis.py`; results in
APPENDIX.md §H. Five public domains; splits as in §5.

- **kNN settings.** k ∈ {1, 5, 10, 20, 50}; Euclidean distance on standardized features (Finding 2 setting) or cosine
  similarity on raw embeddings (Euclidean on L2-normalized vectors). Scored by accuracy and balanced (macro) accuracy;
  the gap uses the converged linear probe scored the same way.
- **Uniformity temperature.** t ∈ {0.5, 1, 2, 5, 10}, all clip pairs, L2-normalized embeddings.
- **Statistic.** Spearman after demeaning within domain; two-sided model-identity permutation p (same permutation in
  every domain, 2,000 draws).
- **Additional transferability estimators.** Shrinkage H-score (Ibrahim et al., 2022): tr(Σ_α⁻¹ (1 − α) Σ_between) with
  Ledoit–Wolf shrinkage (scikit-learn `LedoitWolf`, centred features). GBC (Pándy et al., 2022): −Σ over class pairs of
  the Bhattacharyya coefficient between per-class diagonal Gaussians (variance + 1e-6) on the first 64 principal
  components. All clips, frozen features and labels. Tested against gain exactly as LogME (§7, Finding 3).
- **Similarity measures.** Per domain, 19 × 19 matrices of linear CKA, debiased linear CKA (unbiased HSIC estimator of
  Song et al., 2012, computed in feature space) and mutual k-NN overlap (k = 10, cosine, self excluded; mean over clips
  of the fraction of shared neighbours). Cross-domain stability with the Finding 1 test, BH over the 10 public pairs.
- **Exploratory model-effect removal.** Each matrix double-centred (diagonal excluded) before the same test; centrality =
  Spearman across models of per-model mean similarity between two domains.
- **Non-linear probe head.** `scripts/paper/mlp_converged_gain.py`: the converged-probe gain of §7 with the logistic
  regression replaced by a one-hidden-layer MLP (scikit-learn `MLPClassifier`, 512 ReLU units, Adam, batch 256,
  learning rate 1e-3, ≤ 300 epochs, early stopping on 10% of train with patience 15, `random_state=0`), L2 penalty
  alpha ∈ {1e-4, 1e-2, 1} chosen on validation. Same features (whitened, divided by √d; and a non-whitened variant),
  splits, conditions and checkpoint filter. `scripts/paper/mlp_head_analysis.py` reruns the Finding 3 regression and
  the transferability tests on the MLP gain. Results in APPENDIX.md §H.7.

## 8. Exclusions and missing cells

| Item | Treatment |
|---|---|
| EnCodecMAE | frozen analyses only (no adaptation) |
| Speech, LoRA | some seeds missing (truncated checkpoints); a sensitivity analysis restricted to complete seeds is reported |
| Shared offset | 778 valid adaptation runs (LoRA / top-*K* and ALLoRA); runs whose adapted checkpoint equals the frozen encoder are excluded |
| Music | 3 undecodable FMA-small files dropped for all models |

---

## 9. Software and compute

- **Software.**
  - Python ≥ 3.11. Dependencies are listed in `pyproject.toml`.
  - PyTorch, torchaudio and torchvision are installed separately.
  - Hugging Face `transformers` and `peft` are used for checkpoints and LoRA.
  - `scikit-learn` provides the probes, `scipy` the statistics and `scikit-dimension` TwoNN.
- **Models without Hugging Face checkpoints.** Setup scripts are in `scripts/setup_*.sh`.
- **Execution.** `audio_comp` is used through `PYTHONPATH` (see `README.md`, *Setup*). Jobs ran on Slurm GPU
  clusters; submission templates are in `scripts/slurm/`.
- **Randomness.**
  - Everything is seeded: clip-pool sampling and validation hold-outs use seed 0, adaptation uses seeds 0–4,
    and every permutation and bootstrap uses `numpy.random.default_rng(0)`.

---

## 10. What is and is not distributed

| Artifact | Status |
|---|---|
| Code for extraction, adaptation, geometry and statistics | this repository (`audio_comp/`, `scripts/paper/`) |
| Pretrained checkpoints | public; identifiers and files in `APPENDIX.md` §A and `audio_comp/models/` |
| Adapted model weights | never saved: the adaptation pipeline keeps only the clip × clip RDM and Gram matrix per run |
| Aggregate inputs to every paper analysis (converged-probe accuracies per seed, RSA/CKA matrices, geometry, frozen probes) | `results/` |
| Full results tables and figures | `APPENDIX.md` |
| Clip manifests for the public domains | `data/*_manifest.csv` (the speech speaker manifest is built from LibriSpeech by `xares_eval/librispeech_speaker/build_manifest.py`) |
| Aggregate results for analyses outside the paper | `additional_results/` |
| Raw audio | not redistributed; download from the sources in `README.md` |
| Per-clip RDMs and Gram matrices (≈ 512 GB) | not distributed; regenerated by the adaptation pipeline |
| Any vessel artifact (audio, labels, per-clip or per-seed outputs) | not distributed (§1.3) |
