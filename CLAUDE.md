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
