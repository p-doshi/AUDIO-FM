# Cross-Model Audio Representation Geometry: Draft Write-up

**Status: rough structural draft, 2026-08-11.** Populated with real numbers
pulled from `journal.md`/`results/` throughout — not placeholder prose. Marked
`[OPEN]` where synthesis, a figure, or a Stage 5 result is still needed.
Framing/venue not decided yet (Datasets & Benchmarks floor vs. a fuller
predictive-account ceiling — see CLAUDE.md's Stage 0-6 roadmap); this draft
is written so either framing can be extracted from it without a rewrite.

---

## Abstract [OPEN — draft after Results is locked]

One-line target: independently-trained audio foundation models show partial,
structured agreement on the relational geometry of sound; that structure is
explained by training-distribution breadth better than by domain+paradigm,
by training-*objective* (discriminative vs. reconstruction) better than by
paradigm label, and by an additive domain-relevance effect on top of both —
three findings that each survived a real attempt to falsify them, not just
accumulated. A fourth, methodological finding (alignment/uniformity's
predictive power is scope/task-dependent) is reported as a precise negative
result rather than forced into a false universal claim.

## 1. Introduction / Motivation

- Standard task accuracy can't establish whether a representation is
  well-formed when there's no single correct downstream task to check
  against. This project asks a different question: do independently-trained
  models *agree* on the relational structure of sound (pairwise similarity
  between clips), and if so, what explains the agreement and disagreement.
- Applied framing: **eventual target domain is underwater acoustics**
  (vessel-contact detection) — every model here is a general-audio model,
  not domain-trained; testing whether cross-model consensus geometry
  transfers to a genuinely out-of-distribution domain is the throughline
  from Phase 1 (this write-up) toward Stage 5.
- Related work: Kriegeskorte et al. 2008 (RSA), Kornblith et al. 2019 (CKA),
  Davari et al. 2023 (CKA gameability — load-bearing throughout this
  project's methodology, not just cited), Huh/Cheung/Wang/Isola 2024
  (Platonic Representation Hypothesis — the broader theoretical claim this
  project tests one concrete falsifiable instance of), Wang & Isola 2020
  (alignment/uniformity).

## 2. Methods

### 2.1 Probe set
- 10,000 clips, 5 categories x 2,000 each: music (FMA-small), speech
  (LibriSpeech), bird_sounds, ship_vessel (DS3500/ShipsEar-derived,
  unlabeled), city_noise (UrbanSound8K-derived). Fixed across every model
  and every stage. `data/probe_set_manifest.csv`.

### 2.2 Model roster
- **12 models in the registry** (`audio_comp/models/`, `configs/models.yaml`):
  clap, hubert, wav2vec2, mert, music2vec, musicfm, audio_jepa, beats
  (deferred, loader not wired up), panns_cnn14, bird_mae, ast, audiomae.
- **+1 architectural exception**: birdnet, isolated TensorFlow venv, `.npz`
  embeddings only — see `scripts/birdnet_extract_embeddings.py` and
  CLAUDE.md's Tier 1 table for why it can't live in the normal registry.
- **Checkpoint-provenance discipline** (added 2026-08-10, applies
  retroactively to the whole roster): every model carries a
  `checkpoint_status` (`official_open_weights` /
  `official_public_weights_license_unclear` / `community_conversion` /
  `code_only`), enforced in code at both registration and resolution time,
  not just documented. `bird_mae` is the one `official_public_weights_
  license_unclear` case (no LICENSE file anywhere, confirmed via GitHub's
  license API returning 404). Every paradigm/architecture claim checked
  against the model's primary paper/repo directly, never a secondary
  source — this caught real mislabels twice (music2vec paradigm,
  `beats`'/`audiomae`'s actual license terms) and is now a standing rule in
  CLAUDE.md's Working Conventions, not a one-off habit.
- **Training-distribution-breadth categorization** (the axis that ended up
  mattering most, see Finding 1): narrow-speech (hubert, wav2vec2),
  narrow-music (mert, musicfm), narrow-bioacoustic (bird_mae, birdnet),
  broad-mixed (audio_jepa, clap, panns_cnn14, ast, audiomae),
  self-distillation-narrow-music (music2vec).

### 2.3 Geometry metrics
- RDM: pairwise dissimilarity matrix per model (`correlation` metric),
  10,000 x 10,000, same clip ordering across models.
- RSA (primary metric): Spearman correlation between RDMs' upper triangles.
- CKA (secondary/cross-check only, per Davari et al. 2023 — never
  load-bearing alone; one concrete illustration of why, see Finding 1's
  `birdnet` RSA/CKA divergence).
- TwoNN intrinsic dimension (structural check, `skdim`).
- Alignment/uniformity (Wang & Isola 2020), two implementations:
  same-category proxy (`alignment_score`) and true instance-level positive
  pairs via pitch-shift augmentation (`alignment_score_paired`) — see
  Finding 4 for why the proxy version was replaced.

### 2.4 Downstream functional validation (X-ARES)
- Real toolkit integration (`xares_eval/`), not a custom lightweight probe
  — chosen explicitly for comparability to published baselines (hubert
  validated within a few points of X-ARES's own published wav2vec2
  numbers).
- Tasks: FMA-genre, UrbanSound8K, LibriSpeech-ASR (original 3, 7 models),
  DeepShip (custom private task, vessel-class, 7 models — see 2.5), BirdCLEF
  (custom private task, 50-species, 11 models).
- Both MLP (trained linear/small-head probe) and kNN (untrained,
  raw-embedding-space) protocols reported throughout — the MLP/kNN split is
  itself load-bearing for Finding 2, not incidental.

### 2.5 Domain extension tasks (underwater acoustics preview)
- **DeepShip**: 63-clip GitHub-hosted subset (12 cargo, 20 passengership,
  28 tanker, 3 tug) of the full 265-vessel dataset (full access email-gated,
  not pursued). Real data-integrity finding: the metafile's `record_id` is
  not a reliable join key to hosted filenames (confirmed via duration
  mismatch on ~43% of files) — forced a fallback from vessel-level to
  file-level leakage-control grouping, documented as a weaker-than-ideal
  guarantee, not silently upgraded.
- **BirdCLEF**: `mteb/birdclef25-mini`, 50 species x 20 recordings, clean
  unambiguous metadata (no repeat of DeepShip's join problem) — recording-
  level 5-fold split is the *correct* leakage control here, not a fallback.
- **ShipsEar**: blocked on author email response, not pursued this session.

## 3. Results

### 3.1 Finding 1 — Training-distribution breadth predicts cross-model RSA agreement better than domain+paradigm

- Method: partition models two ways (domain+paradigm vs. breadth), compute
  mean-within-group-minus-mean-between-group RSA, permutation-test the gap
  (20,000 reshuffles, group sizes fixed) given how few models/pairs exist.
- **Result, tracked across three roster sizes as models were added**
  (`audio_comp/pipelines/breadth_hypothesis_check.py`):

  | n models | breadth gap | breadth p | domain+paradigm gap | domain+paradigm p |
  |---|---|---|---|---|
  | 9 | 0.360 | 0.0003 | 0.385 | 0.0095 |
  | 11 | 0.330 | 0.0000 | 0.370 | 0.0082 |
  | 12 | 0.321 | 0.0000 | 0.396 | 0.0063 |

  Breadth's raw gap is consistently *smaller* than domain+paradigm's —
  the win is entirely in significance and coverage. Domain+paradigm's edge
  rests on 2 thin pairs (hubert-wav2vec2, mert-musicfm) that breadth also
  captures identically; breadth additionally explains real structure
  domain+paradigm has no account for at all (the `broad_mixed` 5-way
  cluster's ~10 pairs). At 12 models breadth covers 9-10/12 in real groups
  vs. domain+paradigm's ~5/12.
- **Four independent things pointed at this before it was ever formally
  tested**: the original 10k-scale RSA pattern, audio_jepa aligning with
  clap over same-paradigm models, PANNs' correlations landing on
  audio_jepa/clap, AST's correlations doing the same. Deferred three times
  despite this before actually being run — worth noting in any write-up as
  a methodology point (cheap tests that keep getting deferred are exactly
  the ones worth prioritizing).
- **Known complication, reported not hidden**: `birdnet`'s RSA correlation
  with `bird_mae` (same narrow domain, opposite objective) is only 0.21 —
  much weaker than same-domain-same-objective pairs (hubert-wav2vec2 0.80).
  `audiomae` is similarly a weak fit within `broad_mixed` (0.27-0.52 vs. the
  other four's 0.55-0.84). Breadth explains agreement well in aggregate;
  it does not explain every pair, and the exceptions are informative (see
  Finding 3).
- [OPEN]: Figure — RSA heatmap with breadth-group blocking, or a
  within/between violin plot. `results/rsa_heatmap.png` exists but isn't
  breadth-annotated.

### 3.2 Finding 2 — Discriminative training pressure (not paradigm label) predicts both raw separability and trainable-head exploitability

- Origin: PANNs (supervised, CNN) and Bird-MAE (reconstruction, bioacoustic)
  together suggested "non-contrastive objectives trade raw separability
  (kNN) for trainable-head-exploitability (MLP)" vs. CLAP.
- **AST falsified that reading.** AST is also supervised (AudioSet,
  transformer not CNN) and wins BirdCLEF outright on *both* metrics:

  | Model | MLP | KNN |
  |---|---|---|
  | **ast** | **0.394** | **0.239** |
  | bird_mae | 0.345 | 0.183 |
  | panns_cnn14 | 0.308 | 0.134 |
  | clap | 0.284 | 0.189 |
  | hubert | 0.260 | 0.068 |
  | mert | 0.250 | 0.068 |
  | musicfm | 0.247 | 0.113 |
  | audiomae | 0.167 | 0.039 |
  | music2vec | 0.121 | 0.049 |
  | audio_jepa | 0.089 | 0.043 |
  | wav2vec2 | 0.049 | 0.035 |

  (chance = 1/50 = 0.02; birdnet not yet run through X-ARES, see 2.5/§6)
- **Revised account**: discriminative training pressure — contrastive
  (CLAP) *or* supervised classification with real labels (AST) — produces
  embeddings that are both raw-separable (high kNN, no training) *and*
  trainable-head-exploitable (high MLP); these aren't in tension for
  discriminatively-trained models. Reconstruction training (Bird-MAE,
  AudioMAE) shows the trade-off. PANNs (supervised but CNN, weaker than AST
  on both) suggests architecture may modulate how well supervised pressure
  converts to separable geometry, without being disqualifying.
- Explicitly not a settled account — it replaced the PANNs/Bird-MAE reading
  the same way that reading replaced the original coarse-cohesion framing;
  it should be expected to keep sharpening, not treated as final.
- Also connects back to Stage 1(a)'s original headline finding (coarse
  domain-level cohesion and fine-grained within-domain separability are
  different axes — silhouette-vs-MLP rho=0.357, p=0.43, not significant at
  7 models) — Finding 2 is a mechanistic account of *why* they diverge, not
  a separate claim.

### 3.3 Finding 3 — Domain-relevance is a real, additive effect, isolated via a matched triangle

- **The clean pair**: AudioMAE vs. Bird-MAE, same objective
  (reconstruction-MAE), only domain differs (general AudioSet vs.
  bird-only BirdSet). Bird-MAE wins the bird-specific task by a wide
  margin: MLP 0.345 vs. 0.167 (>2x), kNN 0.183 vs. 0.039 (>4x). This is the
  cleanest isolated domain-relevance test in the project — objective held
  constant, only domain varies.
- **Does not reinstate "domain relevance explains Bird-MAE's original CLAP
  win"** — PANNs (non-domain-matched) already showed the same MLP-wins/
  kNN-loses-to-CLAP pattern Bird-MAE did, before AST recharacterized what
  that pattern actually means (Finding 2). Domain-relevance is additive on
  top of the objective effect, not a competing explanation for it.
- **Third point, same triangle**: BirdNET (discriminative, bird-domain) vs.
  Bird-MAE (reconstruction, bird-domain) — same domain, different
  objective — shows weak mutual RSA agreement (0.21), reinforcing that
  objective matters even holding domain fixed. BirdNET's downstream
  (BirdCLEF/kNN-MLP) numbers are not yet available (§6) — this triangle's
  functional side is currently only 2/3 populated.
- [OPEN]: once BirdNET has a BirdCLEF number, this becomes a full 2x2
  (objective x domain) with functional validation on all four cells, not
  just three geometric ones — worth prioritizing over Stage 5 kickoff if
  the subprocess bridge (§6) turns out cheap.

### 3.4 Finding 4 — Alignment/uniformity's predictive power is scope-dependent, not universal

- Real methodological finding on the way to a diagnostic metric, not a
  failure to find one. Same-category alignment proxy turned out perfectly
  rank-correlated with uniformity (Spearman -1.0, 9 models) — fixed with
  true instance-level positive pairs (pitch-shift augmentation, 500 pairs);
  partial decorrelation achieved (-0.917).
- **The corrected metric's correlation with downstream performance flips
  direction depending on task scope**:

  | Scope (n models) | paired-alignment vs. MLP | paired-alignment vs. KNN |
  |---|---|---|
  | FMA+UrbanSound8K (7) | +0.821, p=0.023 | +0.536, p=0.215 |
  | BirdCLEF (9) | +0.583, p=0.099 | +0.717, **p=0.030** |

  Not resolved by the proxy fix — this is a real property of the metric's
  behavior across task compositions, not an artifact of the degenerate
  proxy (confirmed by re-running with the fixed metric and finding the
  same reversal). **Framed as the actual finding**: which evaluation
  protocol (kNN vs. MLP) alignment/uniformity best predicts depends on
  task composition, in a specific, now-characterized way — more useful to
  a future user of this diagnostic than a false universal claim
  ("uniformity predicts X") would have been.
- [OPEN]: is the scope-dependence explained by which *models* are in each
  scope (7-model roster lacks panns_cnn14/bird_mae/ast/audiomae entirely)
  or by which *task* is being predicted (FMA/UrbanSound8K vs. BirdCLEF)?
  Confounded right now — would need the same 9-11 models run through
  FMA/UrbanSound8K to separate model-roster effects from task effects.
  Bounded, well-defined follow-up if this becomes the Stage 6 diagnostic.

## 4. Discussion / Limitations

- CKA is gameable (Davari et al. 2023) — used only as a cross-check
  throughout; the `birdnet` RSA/CKA divergence (§3.1) is a concrete
  in-project illustration, not just an inherited caution.
- Probe-set domain coverage is general-audio only until real vessel data
  (Stage 5) arrives — DeepShip/BirdCLEF are previews of the OOD question,
  not answers to it (`ship_vessel`/probe-set category itself is unlabeled).
- Small-N statistics throughout (7-12 models) — every permutation
  test/correlation reported with its p-value and n explicitly; several
  results (Finding 4 especially) are honestly reported as underpowered
  rather than oversold.
- BirdNET sits outside the normal registry/checkpoint_status enforcement
  (architectural necessity, not a policy exception) — flag this explicitly
  if BirdNET numbers appear in any figure, so a reader doesn't assume
  uniform provenance-checking across every reported number.

## 5. Stage 5 — OOD vessel fine-tuning

Per CLAUDE.md's roadmap: matched lightweight adaptation (LoRA/small
adapter) on a fixed-size OOD (vessel-acoustic) training slice; measure
adapted performance + few-shot learning curve; correlate against the
frozen-geometry metrics above (breadth-cluster membership, TwoNN ID,
alignment/uniformity). Two fine-tuning conditions (official-released vs.
matched-protocol-across-all-models), only the matched condition used for
the actual scientific comparison.

### 5.1 v1 result (2026-08-11): matched LoRA underperforms frozen probing — a real, if negative, finding

- Ran the cheapest real preview first, per this draft's own earlier
  recommendation: matched LoRA fine-tuning (`audio_comp/pipelines/
  stage5_lora_finetune.py`) on DeepShip's 63-clip subset, 5 models
  (wav2vec2, hubert, mert, music2vec, ast — scoped to models with
  verified-identical LoRA target modules; `panns_cnn14` excluded
  architecturally, pure CNN, no attention layer for a standard LoRA
  target; rest of the roster not yet individually verified).
- **Result**: adapted accuracy is *worse* than frozen-embedding linear
  probing in every comparable case (mert -0.118, wav2vec2 -0.109,
  music2vec -0.013, hubert -0.019 vs. their earlier frozen DeepShip MLP
  scores), and rank-uncorrelated with frozen performance (spearman=0.000,
  n=4). Corroborated by `wav2vec2`'s own 1-epoch smoke test (0.511)
  beating its 10-epoch full-protocol result on the identical fold (0.367)
  — more training made it worse, not better.
- **Diagnosis: overfitting, not a real cross-model adaptability
  difference.** 63 total clips, ~40/fold available for training, one
  class (tug) with only 3 vessels in the whole dataset. This is a
  reportable result on its own (matched lightweight adaptation doesn't
  help at this data scale) but **not yet the geometry-vs-adaptability
  correlation Stage 5 is ultimately for** — deliberately did not force
  the planned breadth-cluster correlation on top of a signal already
  known to be overfitting-dominated (would be noise regressed on noise,
  not a real test of Finding 1).

### 5.2 What v2 needs before the actual question is testable

- Either (a) early stopping against a held-out validation slice or
  stronger regularization (lower LoRA rank, higher dropout/weight decay),
  or (b) more OOD data — which loops back to the still-open ShipsEar
  access request and DeepShip's still-gated full 265-vessel set. (a) is
  cheap and should be tried first; (b) is the actual bottleneck if (a)
  doesn't fix it, since 63 clips may simply be below the floor for any
  fine-tuning protocol to show a real, non-overfit adaptation signal.
- Once a v2 protocol shows *some* real (non-overfit) adaptation signal,
  the original three-way design question is still open and still the
  right one to resolve, in priority order: does breadth-cluster
  membership (Finding 1) predict OOD adaptability (most direct test of
  the project's own motivating question); does the discriminative-vs-
  reconstruction axis (Finding 2) generalize from general-audio
  fine-grained classification to a genuinely OOD domain; does alignment/
  uniformity (Finding 4) predict OOD adaptation where it predicts
  in-domain kNN, separating task-type effects from model-roster effects.

## 6. Deliverables / Reproducibility checklist

- [x] RDM/RSA/CKA/TwoNN toolkit, 12-model + BirdNET roster
- [x] Breadth-hypothesis check, reproducible script
  (`audio_comp/pipelines/breadth_hypothesis_check.py`)
- [x] Alignment/uniformity toolkit (both proxy and true-pair versions),
  reproducible scripts
- [x] X-ARES downstream validation, 3 original tasks (7 models) + BirdCLEF
  (11 models) + DeepShip (7 models)
- [ ] BirdCLEF/DeepShip numbers for `birdnet` (needs a subprocess bridge
  from the PyTorch `xares_eval` venv into the isolated TF venv — scoped
  out of this session, real but bounded work)
- [x] Stage 5 v1 result: matched LoRA underperforms frozen probing,
  overfitting-diagnosed (see §5.1) — real result, not yet the
  geometry-vs-adaptability correlation (see §5.2 for what v2 needs)
- [ ] Figures: RSA heatmap with breadth-group annotation; MLP/KNN scatter
  by training objective; alignment/uniformity scope-dependence plot
- [ ] Decision-rule outcome summary (CLAUDE.md Stage 1(c), still marked
  not started) — this write-up's §3 is most of that summary already,
  worth checking whether Stage 1(c) can just point here rather than
  duplicating
