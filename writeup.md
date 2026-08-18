# ECHO: Cross-Model Audio Representation Geometry — Draft Write-up

**Status: substantially revised, 2026-08-18** (supersedes the 2026-08-11
structural draft — Results and Stage 5 rewritten from scratch to reflect
the current 5-finding state; Methods/Discussion sections carried forward
and updated in place). Populated with real numbers pulled from
`journal.md`/`CLAUDE.md`/`results/` throughout — not placeholder prose.
Marked `[OPEN]` where synthesis, a figure, or a result is still needed.
**Project name: ECHO, used consistently from this revision on** (an
earlier session used "AudioRepBench" in an external discussion only —
that name never actually appeared in this repo's files, so there was
nothing to rename in code/docs, but ECHO is the name to use in any new
writing going forward). Framing/venue not decided yet (Datasets &
Benchmarks floor vs. a fuller predictive-account ceiling — see CLAUDE.md's
Stage 0-6 roadmap); written so either framing can be extracted from it
without a rewrite.

---

## Abstract [OPEN — draft after vessel-domain result lands]

Independently-trained audio foundation models show partial, structured
agreement on the relational geometry of sound. That structure is
explained by training-distribution breadth better than domain+paradigm at
small-to-moderate roster sizes, though this reverses at 18 models —
reported honestly both ways, not smoothed into a single claim. Separately,
training *objective* (discriminative vs. reconstruction, not paradigm
label) predicts both raw embedding-space separability and trainable-head
exploitability for in-domain fine-grained classification, and
domain-relevance is a real, additive effect on top of it. A resolved
alignment/uniformity result (at n=17) shows uniformity predicts kNN
classification more strongly than MLP classification, consistently
across two task scopes. Most directly, the project's central motivating
question — does a model's frozen representational geometry predict its
out-of-domain adaptability — resolves into two precise, confound-checked
answers rather than one: geometry does not predict OOD adaptability on a
near-chance industrial-sound task (MIMII), but embedding-space uniformity
specifically predicts adaptation *headroom* (not final adapted quality)
on tasks with real signal to learn, surviving a parameter-count confound
check and out-of-sample leave-one-out validation. [OPEN: vessel-domain
result, pending.]

## 1. Introduction / Motivation

- Standard task accuracy can't establish whether a representation is
  well-formed when there's no single correct downstream task to check
  against. This project asks a different question: do independently-trained
  models *agree* on the relational structure of sound (pairwise similarity
  between clips), and if so, what explains the agreement and disagreement
  — and does that structure predict anything functionally real.
- Applied framing: **eventual target domain is underwater acoustics**
  (vessel-contact detection) — every model here is a general-audio model,
  not domain-trained. A confidential, AIS-labelled vessel-acoustic dataset
  (streamed only, never copied locally — see CLAUDE.md) is the actual OOD
  test for this question; MIMII (industrial machine sounds) is the public,
  already-completed stand-in that let the core geometry-vs-adaptability
  test be run and validated before the vessel-domain version was ready.
- Related work: Kriegeskorte et al. 2008 (RSA), Kornblith et al. 2019
  (CKA), Davari et al. 2023 (CKA gameability — load-bearing throughout
  this project's methodology, not just cited), Huh/Cheung/Wang/Isola 2024
  (Platonic Representation Hypothesis — the broader theoretical claim this
  project tests one concrete falsifiable instance of), Wang & Isola 2020
  (alignment/uniformity).

## 2. Methods

### 2.1 Probe set
- 10,000 clips, 5 categories x 2,000 each: music (FMA-small), speech
  (LibriSpeech), bird_sounds, ship_vessel (DS3500/ShipsEar-derived,
  unlabeled), city_noise (UrbanSound8K-derived). Fixed across every model
  and every stage. `data/probe_set_manifest.csv`.
- **Known confound, worth stating in any write-up**: probe-set categories
  vary widely in background-noise character (speech/music relatively
  clean and studio-quality; bird_sounds/city_noise/machine_sounds real
  field recordings with natural noise; ship_vessel additionally has
  synthetic channel simulation layered on real recordings) — uncontrolled,
  flagged 2026-08-18. Cross-category RSA differences could partly reflect
  noise robustness rather than purely domain content. Two candidate
  extension sources identified for future noisy-speech/noisy-music
  categories (`ami_meetings`, `singverse_noisy` — see CLAUDE.md), wired up
  but not yet run through `build_probe_set.py`.

### 2.2 Model roster
- **19 models registered, 18 active** as of 2026-08-16 (`audio_comp/models/`,
  `configs/models.yaml`) — grown from the original 12 across two tiers of
  gap-filling additions (AudioMAE, AST, PANNs CNN14, Bird-MAE, EncodecMAE
  for genuinely open representational-paradigm gaps; WavLM, Whisper,
  Data2Vec-audio, MMS, UniSpeech-SAT, SEW, wav2vec2-Conformer as breadth/
  architecture-axis additions). `beats` stays deferred (checkpoint behind
  a manual-download OneDrive link, not a licensing blocker).
- **+1 architectural exception**: birdnet, isolated TensorFlow venv, `.npz`
  embeddings only — see `scripts/birdnet_extract_embeddings.py`.
- **Checkpoint-provenance discipline**, enforced in code at both
  registration and resolution time, not just documented — caught real
  mislabels multiple times (music2vec's paradigm, `beats`'s stale license
  note), now a standing rule (CLAUDE.md Working Conventions).
- **Training-distribution-breadth categorization** (Finding 1): narrow-speech,
  narrow-music, narrow-bioacoustic, broad-mixed, broad-speech,
  self-distillation-narrow-music — see §3.1 for the full grouping and its
  reversal at 18 models.

### 2.3 Geometry metrics
- RDM: pairwise dissimilarity matrix per model (`correlation` metric),
  10,000 x 10,000, same clip ordering across models.
- RSA (primary metric): Spearman correlation between RDMs' upper triangles.
- CKA (secondary/cross-check only, per Davari et al. 2023 — never
  load-bearing alone; the `birdnet` RSA/CKA divergence, §3.1, is a
  concrete in-project illustration of why).
- TwoNN intrinsic dimension (structural check, `skdim`); shows no
  relationship to OOD adaptation gain anywhere it's been tested (§3.5) —
  a real negative result for this specific metric, not a gap.
- Alignment/uniformity (Wang & Isola 2020) — the metric that ended up
  mattering most (Findings 4 and 5). Same-category proxy found perfectly
  rank-correlated with uniformity (Spearman -1.0, one axis not two);
  uniformity is the primary metric reported from this point on.

### 2.4 Downstream functional validation (X-ARES)
- Real toolkit integration (`xares_eval/`), chosen for comparability to
  published baselines.
- In-domain tasks: FMA-genre, UrbanSound8K, LibriSpeech-ASR, BirdCLEF
  (50-species, `mteb/birdclef25-mini`) — all 4 now complete across the
  full 17-18-model roster (grew from the original 3-task/7-model scope).
- Both MLP and kNN protocols reported throughout — load-bearing for
  Finding 2, not incidental.

### 2.5 Matched adaptation protocol (LoRA/ALLoRA), used for both the
in-domain and OOD comparisons
- Two conditions, never blended: **official-released fine-tuned
  checkpoints** (reported separately, answers a different question) vs.
  **matched fine-tuning** (same protocol across every eligible encoder —
  identical split, epochs, optimizer, trainable-layer policy, seeds) —
  only the matched condition is used for any scientific comparison in
  this write-up.
- LoRA coverage: 14 of 18 active models (rank-8, verified per-model
  attention-module naming, not assumed from architecture family).
  **ALLoRA** (Huang & Balestriero, arXiv:2410.09692) added as a
  dropout/scaling-free alternative after LoRA showed a real,
  statistically-verified underperformance-vs-frozen pattern specifically
  on MIMII (see §3.5) — implemented as a custom `torch.autograd.Function`
  with an adaptive per-output-row gradient rescaling, numerically
  validated against gradcheck (for the untouched grad_input) and a
  hand-written reference (for the intentionally-modified grad_A/grad_B).
  Results from LoRA and ALLoRA track each other closely throughout
  (§3.5) — ALLoRA does not change the qualitative story anywhere it's
  been run.
- `panns_cnn14` (no attention layer) and `audio_jepa`/`audiomae`/
  `musicfm`/`encodecmae` (bespoke non-differentiable preprocessing) stay
  excluded from both conditions — real, bounded gap, not silently dropped.

## 3. Results

### 3.1 Finding 1 — Training-distribution breadth predicts cross-model RSA agreement better than domain+paradigm, at small-to-moderate roster sizes — but this reverses at 18 models, reported honestly both ways

- Method: partition models two ways (domain+paradigm vs. breadth), compute
  mean-within-group-minus-mean-between-group RSA, permutation-test the gap
  (20,000 reshuffles, group sizes fixed).
- **Result, tracked across four roster sizes as models were added**:

  | n models | breadth gap | breadth p | domain+paradigm gap | domain+paradigm p |
  |---|---|---|---|---|
  | 9 | 0.360 | 0.0003 | 0.385 | 0.0095 |
  | 11 | 0.330 | 0.0000 | 0.370 | 0.0082 |
  | 12 | 0.321 | 0.0000 | 0.396 | 0.0063 |
  | **18** | **0.2043** | **0.0001** | **0.2394** | **0.0006** |

  At 18 models, domain+paradigm's gap is now *larger* than breadth's — a
  reversal from every prior model count, where breadth's gap led or was
  comparable. Both remain highly significant, and breadth still covers
  more model-pairs in real within-group comparisons (28 vs. 16). Most
  likely explanation: 5 of the 7 Tier-2 additions are narrow-corpus speech
  models that land in essentially the same cluster under *both* framings,
  diluting the specific cases where the two used to disagree. **Per this
  project's own standing gate, breadth can no longer be reported as the
  outright winner at this model count — both are reported together, with
  the reversal stated explicitly, not smoothed over.**
- **Known complication, reported not hidden**: `birdnet`'s RSA correlation
  with `bird_mae` (same narrow domain, opposite objective) is only 0.21 —
  much weaker than same-domain-same-objective pairs (hubert-wav2vec2 0.80).
  `audiomae` is similarly a weak fit within `broad_mixed`.
- [OPEN]: Figure — RSA heatmap with breadth-group blocking at n=18, ideally
  paired with the same figure at n=9-12 to make the reversal visually
  legible, not just tabulated.

### 3.2 Finding 2 — Discriminative training pressure (not paradigm label) predicts both raw separability and trainable-head exploitability

- Origin: PANNs (supervised, CNN) and Bird-MAE (reconstruction,
  bioacoustic) together suggested "non-contrastive objectives trade raw
  separability (kNN) for trainable-head-exploitability (MLP)" vs. CLAP.
- **AST falsified that reading.** AST is also supervised (AudioSet,
  transformer not CNN) and wins BirdCLEF outright on *both* metrics
  (MLP 0.394, KNN 0.239, both best of the roster at the time this was
  established; chance = 1/50 = 0.02).
- **Revised account, held up since**: discriminative training pressure —
  contrastive (CLAP) *or* supervised classification with real labels
  (AST) — produces embeddings that are both raw-separable and
  trainable-head-exploitable, not in tension. Reconstruction training
  (Bird-MAE, AudioMAE) shows the trade-off. PANNs (supervised but CNN)
  suggests architecture may modulate how well supervised pressure
  converts to separable geometry, without being disqualifying.
- **Does this generalize to OOD adaptation gain? Tested directly, §3.5 —
  no.** Objective-type (discriminative vs. reconstruction) does not
  predict LoRA adaptation gain on either MIMII or the higher-signal
  FMA-genre/UrbanSound8K/BirdCLEF tasks (both null, p>=0.29). Finding 2's
  effect is specific to in-domain fine-grained classification skill; it
  does not carry over to adaptation-gain prediction, where uniformity
  (Finding 5) succeeds instead.

### 3.3 Finding 3 — Domain-relevance is a real, additive effect, isolated via a matched triangle

- **The clean pair**: AudioMAE vs. Bird-MAE, same objective
  (reconstruction-MAE), only domain differs. Bird-MAE wins the
  bird-specific task by a wide margin (MLP >2x, kNN >4x) — the cleanest
  isolated domain-relevance test in the project.
- **Does not reinstate "domain relevance explains Bird-MAE's original CLAP
  win"** — PANNs (non-domain-matched) already showed the same pattern
  before AST recharacterized what it means (Finding 2). Domain-relevance
  is additive on top of the objective effect, not a competing explanation.
- **Tested directly as a possible explanation for Finding 5's BirdCLEF
  null (§3.5) — did not hold.** Re-running the uniformity-vs-gain
  correlation on BirdCLEF excluding `bird_mae` moves rho from +0.116 (ns)
  to -0.016 (ns) — collapses toward zero rather than revealing a hidden
  positive relationship. BirdCLEF's category-dependence in Finding 5
  remains a genuine, unexplained exception, not swamped by this effect.

### 3.4 Finding 4 — Alignment/uniformity predicts in-domain kNN more strongly than MLP, resolved at n=17

- Real methodological finding on the way to Finding 5's diagnostic, not a
  side note. Original 9-model/7-model scopes gave a genuinely mixed
  result (uniformity-KNN stronger in BirdCLEF, backwards in FMA+
  UrbanSound8K) — reported as honestly underpowered rather than resolved
  prematurely.
- **Re-run at n=17 once X-ARES coverage gaps were filled for the full
  roster: the mixed result is resolved, not just re-confirmed.** Both
  task scopes now agree: uniformity predicts KNN more strongly than MLP —
  BirdCLEF (uniformity-KNN -0.679, p=0.003 vs. uniformity-MLP -0.583,
  p=0.014) and FMA+UrbanSound8K (uniformity-KNN -0.679, p=0.003 vs.
  uniformity-MLP -0.471, p=0.057). The earlier *backwards* FMA+
  UrbanSound8K result is gone — read as an underpowered n=7 fluke, not a
  real scope disagreement. The strongest, most consistent signal for this
  hypothesis the project has produced.

### 3.5 Finding 5 — Uniformity predicts OOD LoRA-adaptation *headroom* on tasks with real signal to learn; geometry does not predict OOD adaptability on a near-chance task — the project's central motivating question, answered precisely

This is the correlation the project's whole Stage 5/6 design exists to
produce: does a model's frozen general-probe-set geometry predict its
out-of-domain adaptability. Run 2026-08-18, using data that already
existed in `results/` — no new extraction or training needed.

**On MIMII (industrial machine sounds, a near-chance OOD task, 14
LoRA-covered models): a clean null.** TwoNN ID, uniformity, and alignment
each correlated against MIMII LoRA-frozen delta and against raw frozen
MIMII accuracy — all six correlations non-significant (|rho|<=0.30,
p>=0.30). This null survives every confound already separately
identified as a threat to MIMII's other findings (sample-rate bug,
floor-effect near-chance accuracy, DeepShip-style overfitting) — a more
defensible null than a first-pass number. **Power caveat, stated
explicitly**: n=14 with effect sizes this small means "no evidence of a
correlation," not "proof there's no correlation."

**On FMA-genre/UrbanSound8K/BirdCLEF (tasks with real signal to learn,
+0.153 mean LoRA gain, every model positive): a real, confound-checked
positive result.**

| Category | uniformity rho | p | alignment rho | p | TwoNN rho | p |
|---|---|---|---|---|---|---|
| FMA-genre | +0.534 | 0.049 | -0.499 | 0.069 | -0.134 | 0.648 |
| UrbanSound8K | **+0.798** | **0.001** | **-0.776** | **0.001** | -0.257 | 0.375 |
| BirdCLEF | +0.116 | 0.692 | -0.156 | 0.594 | -0.033 | 0.911 |
| Pooled | **+0.736** | **0.003** | **-0.732** | **0.003** | -0.231 | 0.427 |

TwoNN shows no relationship anywhere — specifically a uniformity/
alignment effect. Direction is mechanistically sensible: AST/CLAP
(uniformity approx -2.45, already spread frozen embeddings) have the
least headroom (+0.07 to +0.09 gain); wav2vec2/MMS/Whisper (uniformity
approx -0.08 to -0.18, more collapsed) have the most headroom (+0.18 to
+0.31, Whisper highest). BirdCLEF is a genuine, tested-and-still-
unexplained exception (§3.3).

**Four follow-up checks, each required before this was trustworthy
enough to report as a finding rather than a promising number:**

1. **Uniformity vs. adapted (post-LoRA) accuracy directly, not just
   gain — null** (pooled rho=-0.319, p=0.267). This fixes the claim's
   wording precisely: **uniformity predicts how much a model improves,
   not which model ends up best after fine-tuning.**
2. **BirdCLEF's null re-tested excluding `bird_mae`** — hypothesis
   (domain-relevance swamping headroom) did not hold, see §3.3.
3. **Parameter-count confound — ruled out.** No relationship between
   n_params and uniformity (rho=-0.093, p=0.752) or gain (rho=-0.272,
   p=0.347). Partial correlation of uniformity vs. gain controlling for
   log(n_params): rho=+0.754, p=0.002 — essentially unchanged from raw.
4. **Leave-one-model-out predictive validation** — fit on 13 models,
   predict the 14th, repeat. LOO MAE=0.048 vs. a naive-mean baseline of
   0.060 — a real ~20% out-of-sample error reduction at n=14. Whisper is
   the worst-predicted case (predicted +0.18, actual +0.31), named rather
   than smoothed away — flagged as a hypothesis, not a claim: Whisper's
   ASR/transcript-supervision objective is a different kind of signal
   than AST's classification or CLAP's contrastive pressure, which could
   plausibly leave its frozen representation less "spread" specifically
   on the non-speech domains tested, giving it unusual headroom.

**Objective-type and breadth-cluster tested directly as categorical
predictors of adaptation gain — both null, on both task regimes.**
Discriminative vs. reconstruction (Finding 2's variable): MIMII p=0.291,
pooled FMA/UrbanSound8K/BirdCLEF p=0.885. Breadth-cluster membership
(Finding 1's variable): MIMII Kruskal-Wallis p=0.952, pooled p=0.109.
Neither the variable that explains RSA structure nor the one that
explains in-domain classification skill predicts OOD adaptation gain.

**Before claiming uniformity is a genuinely separate axis from these two
categorical variables — checked whether it's actually independent, since
that determines how strongly this can be stated.** Uniformity is **not**
fully independent of category membership: AST and CLAP are simultaneously
the most uniform models by the continuous metric and the primary drivers
of any apparent categorical effect on uniformity itself (breadth ANOVA
across non-singleton groups: F=9.11, p=0.0087 — but Kruskal-Wallis across
all 6 groups including singletons: p=0.173; collapsed narrow-vs-broad:
p=0.347 — the significance evaporates under more conservative tests, and
objective-type's own group difference on uniformity is similarly
inconsistent, t-test p=0.078 vs. Mann-Whitney p=0.291, driven by Whisper
sitting nowhere near AST/CLAP within its own group). **Despite this
partial overlap, the categorical variables fail to predict adaptation
gain exactly where the continuous metric succeeds — indicating uniformity
preserves resolution among the remaining models that the coarse buckets
discard, rather than simply re-encoding a category label that two
outliers happen to define.** A coarse category can look statistically
real while still being too blunt an instrument to predict adaptation
gain, because a category is driven by its most extreme members while the
continuous metric uses information from every model.

**Net read, worded precisely rather than overclaimed: geometry does not
predict OOD adaptability in general (MIMII null stands on its own terms).
On tasks with real signal to learn, uniformity specifically predicts
adaptation headroom — not final adapted quality — in 2 of 3 categories,
confound-checked and validated out-of-sample. Do not simplify this into
"geometry predicts OOD performance."**

[OPEN — the one remaining empirical gap]: **the confidential vessel-domain
version of this correlation.** This is the domain the project is
actually motivated by; a null on MIMII alone is a weaker result than a
null (or a positive finding) on both. Blocked on: (a) the vessel-domain
matched LoRA fine-tuning run finishing (in progress on a separate,
access-controlled cluster as of this revision — see CLAUDE.md's
`[[confidential_vessel_data]]`), and (b) a user-flagged data-quality
concern that has not yet been diagnosed with specifics. The honest
pre-registered prediction, stated in advance per this project's own
standing discipline: if vessel-domain adaptation behaves like a "real
signal to learn" task (closer to FMA/UrbanSound8K), expect the headroom
effect to reappear; if it behaves like MIMII's near-chance pattern,
expect another null. Either answer completes the OOD story this write-up
is built around — this is not a result waiting to be forced in one
direction.

## 4. Discussion / Limitations

- CKA is gameable (Davari et al. 2023) — used only as a cross-check
  throughout; the `birdnet` RSA/CKA divergence (§3.1) is a concrete
  in-project illustration, not just an inherited caution.
- Probe-set domain coverage is general-audio only until the vessel-domain
  result (§3.5) lands — DeepShip/BirdCLEF/MIMII are previews of the OOD
  question, not full answers to it. DeepShip specifically is deprioritized
  out of active scope as of 2026-08-16 (small, public, and a real
  sample-rate confound in its LoRA pipeline was found and never fixed —
  see CLAUDE.md Appendix A) in favor of the confidential vessel data,
  which has genuinely measured labels and real target-deployment audio.
- Probe-set categories vary widely in background-noise character (§2.1)
  — a real, currently-uncontrolled confound on cross-category RSA
  comparisons, distinct from the cross-model comparisons every finding
  above is actually built on.
- Small-N statistics throughout (7-18 models depending on the analysis)
  — every permutation test/correlation reported with its p-value and n
  explicitly; Finding 5's MIMII null and BirdCLEF exception are both
  explicitly flagged as underpowered-consistent-with-either-answer, not
  oversold as proof of absence.
- BirdNET sits outside the normal registry/checkpoint_status enforcement
  (architectural necessity, not a policy exception) — flagged explicitly
  wherever BirdNET numbers appear, so a reader doesn't assume uniform
  provenance-checking across every reported number.

## 5. Deliverables / Reproducibility checklist

- [x] RDM/RSA/CKA/TwoNN toolkit, 18-model + BirdNET roster
- [x] Breadth-hypothesis check, reproducible at 4 roster sizes including
  the 18-model reversal (`audio_comp/pipelines/breadth_hypothesis_check.py`)
- [x] Alignment/uniformity toolkit, resolved at n=17
  (`audio_comp/pipelines/alignment_uniformity_check.py`)
- [x] X-ARES downstream validation, all 4 in-domain tasks, full roster
- [x] Matched LoRA/ALLoRA-vs-frozen fine-tuning: MIMII (both splits),
  FMA-genre, UrbanSound8K, BirdCLEF — 14-model roster throughout
- [x] The core geometry-vs-OOD-adaptability correlation (Finding 5),
  confound-checked (parameter-count) and out-of-sample validated (LOO)
- [ ] Vessel-domain version of Finding 5 — the one open empirical thread,
  see §3.5
- [ ] BirdCLEF/DeepShip-successor numbers for `birdnet` via X-ARES (needs
  a subprocess bridge from the PyTorch `xares_eval` venv into the
  isolated TF venv — real but bounded, scoped out so far)
- [ ] Figures: RSA heatmap with breadth-group annotation at n=18 (ideally
  alongside n=9-12 for the reversal); MLP/KNN scatter by training
  objective; uniformity-vs-gain scatter for Finding 5, annotated with the
  LOO prediction errors
- [ ] Decision-rule outcome summary (CLAUDE.md Stage 1(c)) — this
  write-up's §3 is most of that summary already; worth checking whether
  Stage 1(c) can just point here rather than duplicating
