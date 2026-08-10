# CLAUDE.md — Cross-Model Audio Representation Geometry & JEPA Relational Distillation

This file orients any Claude session (or collaborator) working in this repo. Read this before touching code.

## One-line summary

Test whether independently-trained audio foundation models agree on the *relational* geometry of sound (pairwise distances between clips), and if they do, test whether a JEPA-style model trained to reproduce that consensus geometry behaves differently — on a real downstream task — than a model distilled from any single teacher.

## Why this project exists

Standard task loss/accuracy can't tell us whether a representation is well-formed when there's no single correct answer to check against. This project is one concrete, falsifiable test of an alternative: does *cross-model agreement on relational structure* track something real, and can that structure be deliberately reproduced rather than just observed. It's the applied, decision-relevant version of a longer theoretical thread (Platonic Representation Hypothesis, CKA/RSA literature, JEPA-as-representation-shaping) — see "Background reading" below if more context is needed, but this file is self-contained for execution purposes.

**Eventual application domain: underwater acoustics** (vessel-contact detection). Do not default to speech/music-only framing when designing the probe set or downstream task — general-audio foundation models are the available *teachers*, not necessarily the target domain.

## Explicit scope

### In scope
- **Phase 1**: Build representational dissimilarity matrices (RDMs) for a fixed probe set of audio clips, one RDM per foundation model, across several models spanning different training paradigms (contrastive, masked-modeling, JEPA-family, generative-derived).
- **Phase 1**: Compare RDMs pairwise via RSA (Spearman correlation between RDMs) and CKA (as a cross-check, not the sole metric — CKA is known to be gameable, see Davari et al. 2023).
- **Phase 2** (gated on Phase 1 findings — see decision rule below): Build a consensus RDM from models that agree, and train a JEPA-style predictor with a relational-distillation objective (reproduce the consensus RDM's structure, not any single teacher's raw embeddings).
- **Phase 2**: Compare downstream task performance (vessel/no-vessel or whatever probe task is finalized) across: (a) consensus-distilled model, (b) single-teacher-distilled models (one per teacher), (c) from-scratch JEPA baseline with no distillation target.

### Explicitly out of scope for this file/repo
- The full compartmentalized acoustic architecture (spectral/modulation/environmental compartments, Kalman-style consolidation) — tracked separately, not part of this experiment.
- SST/oceanographic downscaling work (eddyflow) — unrelated project, do not conflate.
- KAEL / τ-vector continual-identity work — related conceptually, not part of this repo's deliverables.
- Training a foundation model from scratch — infeasible at available compute; this project only uses existing pretrained checkpoints as teachers/probes.

## Decision rule between Phase 1 and Phase 2

Do not start Phase 2 until Phase 1 produces a clear read. Three possible outcomes and what each implies:
1. **Strong cross-model RDM agreement** (most model pairs show high RSA correlation) → proceed to Phase 2 as scoped.
2. **Partial agreement** (some paradigm clusters agree, others don't — e.g. contrastive models agree with each other but not with masked-modeling models) → still proceed to Phase 2, but build the consensus RDM only from the agreeing cluster, and treat the disagreement itself as a reportable finding.
3. **No meaningful agreement** (RDMs uncorrelated across models) → do not proceed to Phase 2 as scoped. This is a legitimate, publishable negative result on its own (evidence against convergence in the audio domain specifically) — pivot to writing this up rather than forcing a distillation experiment onto a null result.

## Extended roadmap (added 2026-08-09, supersedes the Phase 2 sketch above with a staged plan)

Phase 1 reached decision-rule outcome 2 (partial/cluster agreement) at 2000 clips/category, 7 models (see journal.md, 2026-08-09 entries). Rather than jumping straight to the single "Phase 2" experiment originally sketched above (consensus-distilled JEPA vs. single-teacher-distilled vs. from-scratch baseline), the plan is now staged so the project is publishable at a defensible floor even if later stages don't pan out. Contribution framing: the novelty target is understanding of representation space across audio foundation models, not a new architecture or training method. Floor = a Datasets & Benchmarks-track resource (reusable toolkit + curated multi-paradigm roster). Ceiling = a predictive account of what determines representational agreement and functional adaptability, validated against real downstream and OOD fine-tuning performance.

- **Stage 0 — Foundation (complete).** Registry/adapter repo, RDM/RSA/CKA/TwoNN toolkit, `inspect_geometry.py` collapse-vs-idiosyncrasy diagnostic, 7 models running end-to-end on the 10,000-clip probe set. Two classification corrections already load-bearing: music2vec is data2vec-lineage not JEPA (no separate predictor network); audio_jepa (Tuncay et al.) confirmed genuine JEPA architecture, distinct from Fei et al.'s unreleased original A-JEPA.
- **Stage 1 — Consolidate current findings (cheap, do first, no new extraction needed).** (a) **Done, 2026-08-10** — downstream functional probe via real X-ARES toolkit integration (`xares_eval/`) on the 3 overlapping tasks (FMA-genre, UrbanSound8K, LibriSpeech-ASR), all 7 models. **Headline finding (not a footnote to a null result): coarse domain-level cluster cohesion and fine-grained within-domain linear separability are different geometric axes that don't transfer.** This is more specific and more useful than either "geometry predicts function" or "geometry doesn't" — it tells you which geometric property to look for depending on what you're trying to predict, which is exactly what a Stage 6 diagnostic needs to be built on. Evidence (see journal, both 2026-08-10 entries): audio_jepa's coarse-domain cohesion (silhouette 0.60 on our own 5-category probe set, far ahead of every other model — nearly double CLAP's 0.31) does *not* transfer to X-ARES's fine-grained within-domain classification (only mid/lower-pack on both tasks; Spearman rho between our silhouette ranking and X-ARES MLP score = 0.357, p=0.43, not significant). audio_jepa's rank *does* improve modestly under kNN vs. MLP on both tasks, direction-consistent with the source paper's own cohesion/separability claim — real but not dramatic. CLAP's fine-grained dominance (best MLP on both tasks) despite only 2nd-place coarse silhouette points toward contrastive/discriminative training pressure (Wang & Isola alignment/uniformity-style) as a better explanatory variable than raw ID/cohesion — flagged as the stronger Stage 6 diagnostic candidate, not yet formally computed. (b) Breadth-hypothesis check — reframe from domain+paradigm to training-distribution-breadth (narrow-speech: hubert/wav2vec2; narrow-music: mert/musicfm; broad-mixed: audio_jepa/clap; self-distillation-narrow-music: music2vec) and test whether this explains the existing RSA matrix better; gate: if it doesn't explain better, keep domain+paradigm as the working explanation rather than forcing the reframe into the write-up. **Not started.** (c) Written decision-rule outcome summary (outcome 2, now with breadth reframing + downstream-probe result folded in) — this alone is a publishable floor result. **Not started.**
- **Stage 2 — Curated model roster expansion (engineering-heavy, gap-filling not count-filling; formalized as a checkpoint-provenance + tiered plan 2026-08-10, superseding the original single-paragraph sketch below the table).** Add models that test a currently-missing axis, each verified against its primary paper before entering any comparison table (the music2vec lesson applies without exception). Every registered model now carries a `checkpoint_status` field (`audio_comp/models/base.py`) — `official_open_weights`, `official_public_weights_license_unclear`, `community_conversion`, or `code_only` — validated at registration time (`register_model()`) and enforced at resolution time (`get_model_class()` raises if a model without `official_open_weights` or a verified `official_public_weights_license_unclear` status is requested for actual use). This is a real gate, not just a documentation convention: a model can be registered (so the framework already models it) without being usable in `configs/models.yaml`'s `active_models` list until its status is upgraded. All 8 models currently registered (`clap`, `hubert`, `wav2vec2`, `mert`, `musicfm`, `music2vec`, `audio_jepa`, `beats`) are `official_open_weights` — `beats` specifically checked 2026-08-10: unilm's root LICENSE is MIT and the BEATs subdirectory's README defers to it with no separate/contrary statement for the checkpoint weights, read as MIT-covered by absence of any carve-out rather than an explicit per-checkpoint citation (judgment call, documented in `beats.py`'s docstring — re-verify if this becomes a redistribution question). `beats` staying in `deferred_models` is now confirmed to be **purely an engineering gap** (no native `transformers.from_pretrained` path, needs its own loader same as `musicfm`/`audio_jepa` already got) — **not** a checkpoint-availability blocker; worth implementing, not done in this pass.

  **Tier 1 (near-term — each fills a specific, currently-open representational gap):**

  | Model | Checkpoint status | Gap it fills | Status |
  |---|---|---|---|
  | AudioMAE | official_open_weights | Reconstruction-target paradigm (raw spectrogram, not latent) — nothing in the current roster tests this | not started |
  | AST | official_open_weights | Supervised training — tests whether label supervision matters independent of paradigm | not started |
  | PANNs CNN14 | official_open_weights (verified 2026-08-10 against `github.com/qiuqiangkong/audioset_tagging_cnn` directly: MIT LICENSE.MIT, checkpoint on the authors' own Zenodo record) | Pure CNN architecture — every current model is transformer-based; untested ResNet-vs-ViT inductive-bias axis | **wired up + validated 2026-08-10** (`audio_comp/models/panns_cnn14.py`, active in `configs/models.yaml`; interim 9-model RSA/CKA/TwoNN pass and BirdCLEF re-run both complete — see journal for the architecture-axis prediction result, not supported) |
  | BirdMAE | official_public_weights_license_unclear (verified 2026-08-10: no LICENSE file in the GitHub repo — confirmed via GitHub's license API returning 404, not just undetected — and no license field on the HF card; genuinely unresolved, not a formality) | Domain-specific bioacoustic encoder; **also directly serves the `bird_sounds`/BirdCLEF X-ARES extension track (`xares_eval/birdclef/`) — one integration effort, not two** | **wired up + validated 2026-08-10** (`audio_comp/models/bird_mae.py`, active in `configs/models.yaml`; needed a version-skew shim for a transformers 4.38→5.15 incompatibility in the checkpoint's own vendored code — see module docstring; BirdCLEF re-run shows a real but nuanced result, see journal — beats CLAP on MLP, not on KNN, and PANNs shows the identical pattern despite no domain match, complicating a pure domain-relevance explanation) |
  | Perch 2.0 | official_public_weights_license_unclear — verify before use | Second independent bioacoustic-domain check; same dual-purpose note as BirdMAE | not started |
  | EnCodecMAE | official_open_weights | Neural-codec-derived representation — discrete/compressed representation type, distinct from BEATs' self-distilled tokenizer approach | not started |

  **Tier 2 (lower priority — doesn't fill a currently-open gap, defer until Tier 1 is done):**

  | Model | Checkpoint status | Note |
  |---|---|---|
  | WavLM | official_open_weights | Another speech masked-modeling variant — roster already has 2 (hubert, wav2vec2) |
  | BYOL-A | official_open_weights, fine-tuned variants inconsistent | BYOL without masking — useful but secondary to AudioMAE/PANNs |
  | Whisper | official_open_weights | Better suited to a separate speech-specialized analysis than the main paradigm comparison |
  | NatureLM-audio | official_open_weights | Multimodal audio-language model — **needs an explicit written decision on what layer/pooling defines "the representation" before it can enter an RDM comparison at all**, not a plain encoder like the rest of the roster |

  **Gate for every addition, Tier 1 or 2, before any adapter code is written:** weights downloadable without private access; license permits research use; produces deterministic embeddings through a documented adapter; checkpoint_status verified against the model's **primary paper/repo directly, not a secondary source** — this is a hard requirement given the music2vec and A-JEPA/Audio-JEPA mix-ups already hit in this project, not optional diligence. Gate before Stage 3 continues to apply on top of this: every new adapter smoke-tested on short/edge-case clips (per the audio_jepa kaldi.fbank lesson), checked for dependency-version side effects (per the flash_attn/torch lesson).
- **Stage 3 — Re-run the geometry toolkit at ~11 models.** Full RDM/RSA/CKA/TwoNN sweep on the same fixed probe set; re-test the breadth hypothesis with the new axes; re-run `inspect_geometry.py` per new model individually. **This is the point at which the Datasets & Benchmarks-track floor is fully met.** **Interim 9-model pass done 2026-08-10** (`panns_cnn14` + `bird_mae` only, not the full Tier 1 roster) — not the formal Stage 3 completion (that's ~11 models, 4 more Tier 1 additions still needed), but real signal already: PANNs' RSA agreement with the transformer-based models (mean 0.334) is *higher*, not lower, than the transformers' agreement with each other (mean 0.283) — no evidence architecture family is an independent geometry-shaping axis in this data, a clean negative result for that specific hypothesis. PANNs' two highest individual correlations (`audio_jepa` 0.60, `clap` 0.56) line up with the still-unstarted Stage 1(b) breadth-hypothesis grouping (broad-training-distribution models) better than paradigm or architecture do — second independent point in that reframing's favor.
- **Stage 4 — Functional validation at the expanded roster.** Repeat Stage 1's downstream probe across all ~11 models; test whether the surviving explanatory variable predicts downstream accuracy, not just mutual RSA. **Interim check done 2026-08-10**: re-ran BirdCLEF (not the full 3-task X-ARES suite) with `panns_cnn14`/`bird_mae` added. BirdMAE beats CLAP on MLP (0.345 vs. 0.284) but not KNN (0.183 vs. 0.189) — real, but PANNs (no bioacoustic domain match at all) shows the *identical* MLP-wins/KNN-loses relationship to CLAP, which undercuts a pure domain-relevance explanation for BirdMAE's MLP edge. Sharper reading: CLAP's contrastive embedding space is more separable untrained (KNN), while PANNs' supervised and BirdMAE's reconstruction-MAE objectives both produce features a trained MLP head can exploit more, regardless of domain match. A genuine extension of the Stage 1(a) cohesion-vs-separability headline finding — now a three-way split (coarse cohesion / fine-grained separability / raw-separability-vs-trainable-head-exploitability), not fully resolved with only 9 models and 4 tasks. Full log in journal.md, 2026-08-10 entries.
- **Stage 5 — OOD fine-tuning extension (first half of the novelty ceiling).** Target: real vessel/underwater-acoustic data (genuinely OOD for every current model, and the actual eventual application domain — DeepShip's 63-clip GitHub subset, see journal 2026-08-10, is an early real-data preview of this axis, not yet the formal Stage 5 run). Matched lightweight adaptation (LoRA/small adapter, not full fine-tuning) on a fixed-size OOD training slice; measure final adapted performance and the few-shot learning curve; correlate against Stage 3's frozen-geometry metrics (ID, cohesion/separation index, RSA-cluster membership). Note for any write-up: this tests representation-*space* intrinsic dimension predicting adaptability — a different object from Aghajanyan/Gupta/Zettlemoyer (ACL 2021)'s fine-tuning *parameter-update* intrinsic dimension finding; same term, different space, state explicitly. Falsification is a legitimate outcome here too: no correlation just means geometry is descriptive but not predictive of adaptability.

  **Fine-tuning methodology (formalized 2026-08-10): two separate conditions, never blended.**
  1. **Official fine-tuned condition.** Use released fine-tuned checkpoints where available. Log, per model: fine-tuning dataset, which layers were updated, number of output classes, the checkpoint-selection rule (e.g. best-val-epoch vs. final), and whether the encoder itself was tuned or only a task head on top of a frozen encoder. Report this condition **separately** — it answers "how good is the model's own best publicly-released adaptation," a different question from the matched condition below, and mixing the two into one comparison table would silently conflate training-recipe differences with representation-geometry differences.
  2. **Matched fine-tuning condition — this is the only one used for the main scientific comparison** (correlating frozen-representation geometry against OOD adaptability, per the paragraph above). Fine-tune every eligible encoder under one identical protocol: same downstream dataset, same train/val/test split, same number of epochs, same optimizer/scheduler, same trainable-layer policy (e.g. LoRA rank and target modules, or which layers are unfrozen), same seeds. Only comparable across models because every axis except the frozen starting representation is held fixed — the entire point of Stage 5's falsifiable claim depends on this.
- **Stage 6 — Practical predictive diagnostic (the actual novelty ceiling).** Only if Stage 5 finds a real correlation: build a lightweight tool that, given a new unseen foundation model, computes RDM/ID/cohesion on a small unlabeled probe set (no fine-tuning, no labels) and predicts likely OOD adaptability + nearest existing cluster, via the relationship learned across the Stage 3 roster. This is what would justify a venue beyond the Datasets & Benchmarks floor. **Candidate diagnostic metric, motivated by Stage 1's actual result (2026-08-10):** raw TwoNN intrinsic dimension / coarse silhouette did not predict X-ARES fine-grained classification performance, but CLAP's specific pattern (best fine-grained classifier, only 2nd-place coarse cohesion) is consistent with contrastive/discriminative training pressure being the more relevant variable. Worth computing real Wang & Isola alignment/uniformity scores (not just the silhouette proxy used in Stage 1) alongside TwoNN ID for the Stage 3 roster, and testing whether *that* correlates with Stage 5's OOD adaptability results where raw ID doesn't.

**Cross-stage risks to restate at every stage, not just once:** CKA is gameable (Davari et al. 2023) — never load-bearing alone. Compute/training-scale confounds apply to every new model in Stage 2, same as the audio_jepa undertraining confound. Probe-set domain coverage is general-audio only until Stage 5's OOD data arrives — don't claim domain generality beyond what's tested. Aggregator/secondary-source labels are not verification — every paradigm/architecture claim gets checked against its primary paper (the music2vec correction is the standing example of why).

## Candidate teacher models (starting list)

Pull individual pretrained checkpoints (mostly HuggingFace) — no single unified benchmarking codebase exists for this list, it will need to be assembled by hand. Aim for paradigm diversity, not just model count:

| Model | Paradigm | Notes |
|---|---|---|
| CLAP | Contrastive (audio-text) | |
| MERT | Masked modeling (music) | |
| HuBERT / wav2vec 2.0 | Masked modeling (speech) | |
| BEATs | Masked modeling | |
| A-JEPA | JEPA-family (online distillation) | Already JEPA-paradigm — useful within-paradigm comparison point |
| music2vec | data2vec-family (self-distillation, EMA-updated teacher) | **Not JEPA-family** — see correction below. Originally mislabeled here as JEPA-family. |
| MusicFM | Masked modeling (BEST-RQ) | |

Do not treat this table as final — confirm checkpoint availability and license before committing to any model.

**Correction (2026-08-09):** music2vec was originally labeled "JEPA-family" in this table and that framing carried through several turns of Phase 1 analysis, including treating its cross-model isolation as informative about H1. That doesn't hold up. music2vec (data2vec-style) has a student encoder that operates directly on masked input and predicts the EMA teacher's averaged top-K layer representations — no separate predictor network. JEPA as specified (and as A-JEPA/Audio-JEPA's own methods sections describe) has three distinct components: a context encoder, an EMA-updated target encoder, and a **separate predictor network** P_φ conditioned on the context representation, trained specifically to predict target-region representations. Both lineages descend from BYOL and are self-distillation-with-EMA-teacher, but "has a decoupled predictor network" is the actual dividing line, and music2vec is on the data2vec side of it. Practical consequence: as of this correction, there are **zero working JEPA-family models** in the active comparison (music2vec is data2vec-family; the original A-JEPA has no public checkpoint; the substitute `ltuncay/Audio-JEPA` is registered but not yet wired up — see `audio_comp/models/audio_jepa.py`). Once `audio_jepa` is wired up it will be the *first* JEPA-family data point, not a second one to compare against music2vec — H1 as originally phrased (mutual agreement *among* JEPA-family models) is untestable until a second working JEPA-family checkpoint exists.

**Confound to flag before `audio_jepa` goes into any comparison:** the `ltuncay/Audio-JEPA` substitute is trained on meaningfully less compute than the other active teachers — 100k steps (~14h on 4 V100s, 5,338h of AudioSet) vs. wav2vec2/data2vec's 400k steps on larger batches. Its own paper reports it substantially underperforming both baselines on several linear-probe tasks (e.g. Speech Commands V1: 0.152 vs. data2vec's 0.927). If `audio_jepa` comes back RSA-isolated from the other five once wired up, that's confounded between two explanations that matter for very different reasons — "JEPA-paradigm geometry is genuinely different" (the thing this project wants to test) vs. "this particular checkpoint is comparatively undertrained" (unrelated to the paradigm question) — and RSA alone can't distinguish them, the same way it couldn't for music2vec's isolation (see `audio_comp/pipelines/inspect_geometry.py` and the 2026-08-09 journal entry for how that was disambiguated).

**A specific, testable prediction for when `audio_jepa` is wired up:** the Audio-JEPA paper states its objective favors embedding cohesion over linear separability — strong kNN performance alongside weak linear-probe performance on the same tasks. That predicts a checkable geometric signature: run `inspect_geometry.py` on `audio_jepa` once live and check for a *low* TwoNN intrinsic dimension and/or unusually tight within-category clustering relative to the other five. Note this is the **opposite direction** from music2vec, which had the *highest* intrinsic dimension of the six active models. If `audio_jepa` shows low ID/high cohesion, that's a real, paper-grounded basis for why JEPA's geometry might differ from the masked-modeling clusters — much stronger evidence than an unexplained isolated RSA number.

## Probe set requirements

- Must include enough clips to estimate RDMs reliably (published cross-model RSA/CKA work typically uses thousands of samples for global comparisons — smaller sets are noisier).
- Should span the domain the downstream task will eventually test (i.e., include underwater/vessel-relevant sounds if available, not only speech/music), since general-audio FMs are not guaranteed to organize non-speech/music sound sensibly — this is itself part of what Phase 1 is testing, not an assumption to bake in.
- Fixed across all models for the whole experiment — do not vary the probe set per model.

## Metrics

- **RSA** (Spearman correlation between RDMs) — primary metric.
- **CKA** — secondary/cross-check only. Do not report CKA alone; pair with RSA or an ID-based diagnostic.
- **Intrinsic dimension (TwoNN)** per model, as an independent structural check — useful for interpreting *why* two models might disagree (e.g. one collapsed to low ID).

## Expected results — stated as falsifiable predictions, not desired outcomes

- H1: JEPA-family models (A-JEPA, music2vec) show higher mutual RDM agreement with each other than with non-JEPA-family models, at matched probe set. **[Correction 2026-08-09: music2vec is data2vec-family, not JEPA-family — see correction note above. H1 as phrased needs a second working JEPA-family checkpoint before it's testable; currently only the `audio_jepa` substitute qualifies, and it isn't wired up yet, so this is a single data point, not a within-paradigm comparison.]**
- H2: Consensus-distilled model (Phase 2) shows downstream performance at least as good as the best single-teacher-distilled model, if H1's convergence signal is real and functionally meaningful.
- Falsification of H2 (consensus underperforms every single teacher) is itself a valid, reportable outcome — do not treat it as an experiment failure requiring rework; it's evidence that idiosyncratic per-model structure carried functional signal that averaging destroyed.

## Deliverables

1. RDM matrices + RSA/CKA comparison heatmaps across all teacher models (Phase 1).
2. Written summary of which decision-rule outcome (1/2/3 above) was reached, before any Phase 2 code is written.
3. (If Phase 2 proceeds) Trained consensus-distilled model, single-teacher-distilled models, and from-scratch baseline, plus downstream task comparison table.
4. Short report suitable for a workshop submission (Interspeech/ICASSP/bioacoustics venue) or as a section of a larger thesis chapter — framing depends on which decision-rule outcome was reached.

## Known risks / limitations to keep stated in any write-up

- CKA is gameable (Davari et al., 2023) — never the sole evidence for a claim.
- Most candidate teacher models are speech/music-trained, not underwater-acoustic-trained — any convergence found may not transfer to the eventual application domain, and this should be tested explicitly, not assumed.
- RDM/RSA comparisons are sensitive to probe-set composition and size — report probe-set construction details explicitly in any write-up, and treat small-probe-set results as preliminary.
- A negative Phase 1 result is a valid stopping point, not a failure — do not force Phase 2 onto a null result to "have something to show."

## Background reading (for context, not required to start Phase 1)

- Kriegeskorte et al. 2008 — RSA / representational dissimilarity matrices, the core method for Phase 1.
- Kornblith et al., ICML 2019 — CKA.
- Davari et al., ICLR 2023 — CKA gameability limitation.
- Park et al., CVPR 2019 — Relational Knowledge Distillation, closest precedent for Phase 2's distillation objective.
- Huh, Cheung, Wang & Isola, ICML 2024 — Platonic Representation Hypothesis, the broader theoretical claim this project tests a concrete instance of.
- Sengupta, "Representation Without Reward: A JEPA Audit for LLM Fine-Tuning," arXiv 2605.15394 (May 2026) — closest precedent for testing whether representation-level objectives produce decoder/task-visible change, methodology template for Phase 2's evaluation design.

## Working conventions

- State assumptions explicitly in code comments and commit messages — this project has several open design choices (probe set composition, consensus-RDM construction method, downstream task selection) that are not yet finalized.
- Do not silently expand scope into the excluded-topics list above without discussing first.
- Prefer reproducible, seeded runs — cross-model comparisons are only meaningful if noise from randomness is distinguishable from genuine disagreement.
- **Verify any factual claim about a model — paradigm, architecture, checkpoint provenance, license — against its primary paper or repo directly before it enters code, a comparison table, or a write-up, even when the claim comes from an instruction, a prior note in this file, or your own earlier output in the same session.** A label someone (including a past version of you) already wrote down is not verification; it's a claim that hasn't been re-checked. This is a standing rule, not a one-off reaction to a single incident — it's what caught music2vec's paradigm mislabel (2026-08-09), the A-JEPA/Audio-JEPA substitution needing to stay explicitly flagged, and `beats`' license note being stale relative to the actual unilm LICENSE file (2026-08-10, caught specifically by re-checking rather than applying an instructed label at face value). Cost of checking is a few minutes; cost of an unverified claim propagating into a comparison table is a silent, hard-to-trace correction later.

## journal.md — mandatory, ongoing

Maintain `journal.md` at the project root as a running daily log. This is
not optional and not just for milestones — **log continuously, including
failures and small steps**, not only completed features.

Every time you (Claude) do any of the following, append an entry:
- Try an approach that fails, including *why* it failed if known
- Make a small implementation change (not just big features)
- Discover something about the data (buoy quirks, artifacts, surprises)
- Make a design decision, including ones later reversed
- Finish a meaningful chunk of work

Entry format — append, never rewrite history:

```
## YYYY-MM-DD

- [what was tried / done] — [outcome] — [why, if a failure or a decision]
```

Keep entries terse and factual — this is a lab notebook, not a report.
The goal is that six months from now, either of us can reconstruct *why*
a decision was made or *what* was already tried and ruled out, without
re-deriving it. When in doubt about whether something is worth logging,
log it — the cost of an extra line is near zero; the cost of losing a
dead end that gets re-tried later is not.
