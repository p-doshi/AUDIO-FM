
# Audio Comp — Findings Index

Retrieval index over `journal.md` (574 lines, 2026-08-08 → 2026-08-19) and `CLAUDE.md`. Organized by project area, chronological within each. Numbers are copied verbatim from source; discrepancies between CLAUDE.md and journal.md are flagged where found. This is not an editorial pass — nothing is dropped for seeming minor.

---

## 1. Main RSA/CKA pipeline (roster growth, RSA/CKA/TwoNN, breadth hypothesis)

**2026-08-08 — pilot (100 clips, 20/category, 6 models: clap/hubert/mert/music2vec/musicfm/wav2vec2).**
- hubert↔wav2vec2 (speech, masked-modeling) RSA 0.80, CKA 0.83. mert↔musicfm (music, masked-modeling) RSA 0.49, CKA 0.52. Cross-domain/cross-paradigm pairs mostly RSA 0.1–0.3.
- music2vec (sole active JEPA-family label at the time) max RSA 0.25 (with musicfm) — doesn't correlate strongly with anything.
- Read against decision rule: outcome 2 (partial/cluster agreement), but flagged as pilot-only, not a real Phase 1 read.

**2026-08-08 (cont'd) — scaled to 2000/category (10,000 clips), same 6 models.**
- hubert↔wav2vec2 RSA 0.80; mert↔musicfm RSA 0.51. Cross-domain/cross-paradigm pairs 0.02–0.36.
- music2vec still isolated: max RSA 0.25 (musicfm) — same as pilot.
- clap↔musicfm: RSA 0.36 but CKA 0.67 — concrete RSA/CKA divergence example.
- TwoNN ID: wav2vec2/mert lowest (~11.5), music2vec highest (~21.5).
- `inspect_geometry.py` built to test music2vec collapse-vs-idiosyncrasy: NOT collapsed. Within-category < between-category distance for all 5 categories, separation index 0.54 (hubert reference 1.02, but hubert's number is driven by a speech-vs-everything gap). Pairwise-distance CV=0.43 (not ~0, rules out collapse). Combined with highest TwoNN ID (21.5) — collapse shows as *low* dimension, so this also argues against collapse.

**2026-08-09 — music2vec paradigm correction.** music2vec reclassified data2vec-family, not JEPA-family (no separate predictor network — see §6 for full provenance detail). Zero working JEPA-family models exist in the comparison until `audio_jepa` is wired up.

**2026-08-09 (cont'd) — audio_jepa wired up, 7 models now active.**
- TwoNN ID: audio_jepa **10.82**, lowest of 7 (range 11.5–21.5 for the rest; music2vec highest 21.5).
- `inspect_geometry.py`: within-category mean distance **0.062** (audio_jepa) vs hubert 0.256, music2vec 0.161. Diagonal 0.02–0.06 for bird/city/ship/speech; music an outlier (0.42–0.50 from other categories). Separation index 1.36 (highest of the three models checked this way).
- RSA: audio_jepa↔clap = **0.54** (strongest pair in the 7-model matrix, edging out mert↔musicfm's 0.51), ↔musicfm 0.42, ↔mert 0.24, ↔music2vec 0.20, ↔hubert 0.02, ↔wav2vec2 **-0.08**. CKA same pattern (clap 0.50 highest, wav2vec2 0.13 lowest). NOT RSA-isolated the way music2vec is.

**2026-08-10 — 9-model RSA/CKA/TwoNN pass (adds panns_cnn14, bird_mae).**
- Architecture-axis prediction (PANNs vs transformers): **not supported**. Mean RSA among 8 transformer models (28 pairs) = 0.2826; mean RSA panns_cnn14↔those 8 = 0.3336 — PANNs agrees *more* with the transformer roster than the transformers agree with each other.
- PANNs' two highest individual correlations: audio_jepa 0.60, clap 0.56 — early signal for the breadth-hypothesis reframing (below).
- bird_mae RSA pattern flat/uniform (0.28–0.45), no sharp outlier.
- TwoNN: panns_cnn14 12.524, bird_mae 14.348 (both unremarkable).

**2026-08-10 (cont'd) — Stage 1(b) breadth-hypothesis check, first run (9 models).**
- Two framings tested: domain+paradigm vs. training-distribution-breadth (narrow-speech / narrow-music / narrow-bioacoustic / broad-mixed / self-distillation-narrow-music).
- domain+paradigm: gap = 0.385 (mean_within 0.658, only 2 real pairs), permutation p = 0.0095.
- breadth: gap = 0.360 (mean_within 0.604, 5 pairs), permutation p = **0.0003**.
- Verdict: breadth adopted as working explanation — smaller raw gap but explains far more of the roster (7/9 models in real groups vs 5/9) with a stronger significance signal.

**2026-08-10 (cont'd) — 11-model re-run (adds ast, audiomae).**
- breadth permutation p = **0.0000** (0/20,000 reshuffles matched), domain+paradigm p = 0.0082. Breadth's raw gap (0.330) now slightly below domain+paradigm's (0.370), but covers 9/11 models in real groups vs 4/11.
- broad_mixed cluster now 5 members (audio_jepa, clap, panns_cnn14, ast, audiomae); audiomae's own fit within the group is weaker (RSA 0.27–0.52 vs the other four's mutual 0.55–0.84) — reported, not smoothed over.

**2026-08-10 (cont'd) — 12-model re-run (adds birdnet, isolated-environment exception, see §6).**
- breadth permutation p stays **0.0000**; raw gap drops to 0.321 (domain+paradigm rises to 0.396, noise at n=2 pairs).
- birdnet's RSA fit weak across roster: max 0.32 (musicfm), one negative (-0.12 with audiomae). birdnet↔bird_mae (same narrow_bioacoustic domain, opposite objective) only **0.21** — much weaker than same-domain-same-objective pairs (hubert-wav2vec2 0.80, mert-musicfm 0.51) — real evidence objective shapes geometry even within fixed domain.
- CKA cross-check diverges sharply: birdnet's CKA with the roster is 0.24–0.66 (vs RSA 0.01–0.32) — concrete RSA/CKA divergence, not resolved either way (CKA-secondary rule).
- TwoNN: birdnet 17.9 (2nd-highest of 12, behind music2vec's 21.5).

**2026-08-12 — 18-model RSA/CKA/TwoNN sweep** (job 54519498, 1h28m).
- Headline: data2vec_audio↔music2vec (first same-paradigm/cross-domain pair, data2vec-family) RSA **0.086** — much weaker than same-domain-same-objective pairs (hubert-wav2vec2 0.80).
- MMS moderate correlation with other speech models (0.37–0.52, strongest with unispeech_sat 0.52).
- TwoNN new-model range: sew 10.43 (lowest in whole roster), wav2vec2_conformer 18.54 (2nd-highest after birdnet's 17.87). Not yet interpreted at time of logging.
- Breadth-hypothesis not yet re-tested at 18 models in this entry (done 2026-08-13, below).

**2026-08-13 — breadth-hypothesis re-run at 18 models: reversal.**
- New buckets: `data2vec_speech`/`data2vec_music` split (domain+paradigm); wavlm/unispeech_sat/sew/wav2vec2_conformer join `masked_modeling_speech`/`narrow_speech` (same 960h LibriSpeech corpus); mms/whisper get new `broad_speech` bucket (domain-narrow, distribution-broad).
- **domain+paradigm gap (0.2394) now larger than breadth's (0.2043)** — reversal from every prior model count. Both still highly significant (domain+paradigm p=0.0006, breadth p=0.0001). Breadth still covers more pairs (28 vs 16).
- Verdict per standing gate: breadth "can no longer be reported as the outright winner" at 18 models — report both going forward, state the reversal explicitly.

**2026-08-18 — `rsa_by_category.png` (per-category RSA, 18×18, 5 categories, final-layer only, one-off sbatch).**
- `data2vec_audio` reads as isolated dark row/column in nearly every category (consistent with its 0.086 vs music2vec).
- `audiomae` isolated specifically in `speech`.
- Warm-colored cluster (hubert/mert/mms/musicfm/panns_cnn14/sew/unispeech_sat) co-clusters in music/city_noise/ship_vessel. Not formally tested, visual pattern only.

**2026-08-13 — `ship_vessel` (DS3500) data-quality gap found.** ~9.6% of DS3500 is "Class E, environmental noise" (no vessel present); traces to only 1,948 real ShipsEar recordings expanded via synthetic ray-theory channel simulation; distance-to-vessel metadata (1–11km) is simulated, not measured. Logged in CLAUDE.md Known Risks as its own subsection. Should have been checked at Phase 1 kickoff — wasn't.

**2026-08-18 — background-noise character audit across probe-set categories (raised by user question).** speech (LibriSpeech) and music (FMA-small) are clean studio audio; bird_sounds (Xeno-Canto)/city_noise (UrbanSound8K)/confidential vessel data are real-world field recordings; machine_sounds (MIMII) deliberately noisy (only 6dB SNR tier used — MIMII's cleanest tier, still substantial noise); ship_vessel (DS3500) has synthetic channel simulation on top of real recordings. Flagged as an uncontrolled confound between "noise robustness" and "domain/paradigm" in cross-category RSA comparisons.

**2026-08-18 — `music_noisy` (SingVERSE) sampling-fraction caveat.** Standard 2,000-clip draw is ~51% of SingVERSE's entire 3,929-clip corpus (every other category draws <25%). Kept at 2,000 for parity anyway; flagged possible singer/song non-independence.

**2026-08-18 — AMI/SingVERSE noise-robustness prediction, pre-registered before extraction.** Models with broad/noisy training exposure (whisper, mms, audio_jepa) predicted to show *smaller* RSA/CKA shift between clean/noisy versions of the same content (speech vs speech_noisy via ami_meetings; music vs music_noisy via singverse_noisy) than narrow/clean-corpus models (hubert, wav2vec2). Not yet run as of 2026-08-19 (sources wired up, not yet through `build_probe_set.py`).

---

## 2. X-ARES downstream validation (Stage 1 / Stage 4)

**2026-08-10 — Stage 1 X-ARES run, 3 tasks (FMA-genre, UrbanSound8K, LibriSpeech-ASR), 7 models.**
- Validation vs. published numbers: hubert FMA MLP 0.482 vs published wav2vec2 baseline 0.469; UrbanSound8K MLP 0.682 vs published 0.659.
- Full MLP table: clap 0.687/0.888/0.000; musicfm 0.615/0.799/0.000; mert 0.570/0.774/0.000; audio_jepa 0.552/0.556/0.000; music2vec 0.521/0.601/0.020; hubert 0.482/0.682/0.855; wav2vec2 0.185/0.226/0.000 (columns: FMA/UrbanSound8K/LibriSpeech-ASR).
- Initial reading: "geometry doesn't predict function" — audio_jepa (lowest ID) only mid-pack; CLAP (mid ID) best classifier on both tasks.
- LibriSpeech-ASR returns 0.0 for everyone except hubert (0.855) and music2vec (0.02) — wav2vec2's 0.0 flagged as unexplained (see §7 for the investigation).

**2026-08-10 (cont'd) — refinement of the null (user-pushed, 3 checks).**
1. kNN vs MLP rank check: audio_jepa's *relative rank* improves under kNN (FMA 4th→3rd, UrbanSound8K 6th→5th) — real, direction-consistent, modest. kNN/MLP ratio highest for CLAP (0.94), not audio_jepa (0.69, 3rd) — argues against ratio as a clean cohesion indicator.
2. Silhouette score (own 5-category probe set, all 7 models): audio_jepa **0.602**, nearly double CLAP's 0.313 (2nd). Correlated against X-ARES avg MLP (FMA+UrbanSound8K): Spearman rho = **0.357, p = 0.432** — weak, not significant (n=7). Refined finding: coarse-domain cohesion ≠ fine-grained within-domain separability — different geometric axes.
3. wav2vec2 wiring diff: HubertXaresEncoder/Wav2Vec2XaresEncoder measured identical (sampling_rate=16000, output_dim=1024, hop_size_in_ms=20.101). do_normalize/feat_extract_norm/do_stable_layer_norm all identical between hubert and wav2vec2. X-ARES's own collate discards the lengths tensor before any encoder call (confirmed via source read) — real but non-differentiating (hubert shares the same risk and scored 0.855 anyway). **Downgraded from "possible bug" to "confirmed not a wiring bug, cause unknown."**

**2026-08-10 (cont'd) — DeepShip task (63 clips, 7 models, 3-fold, file-level split — see §3 for split-integrity detail).**
- MLP/KNN: mert 0.494/0.337; musicfm 0.430/0.386; wav2vec2 0.415/0.310; music2vec 0.408/0.294; audio_jepa 0.402/0.288; clap 0.369/0.325; hubert 0.364/0.355. Chance = 0.25.
- Two music-domain models (mert, musicfm) take 1st/2nd MLP; musicfm wins KNN outright — flagged as a candidate pattern (n=2/7, 1 dataset), not a strong claim.
- audio_jepa 5th/7 MLP, last/7 KNN — consistent with (not proof of) its 20Hz feature-extraction low-freq cutoff (see §6).

**2026-08-10 (cont'd) — BirdCLEF task (1000 clips, 50 species, 5-fold, 7 models).**
- MLP/KNN: clap **0.284/0.189**; hubert 0.260/0.068; mert 0.250/0.068; musicfm 0.247/0.113; music2vec 0.121/0.049; audio_jepa 0.089/0.043; wav2vec2 0.049/0.035. Chance = 0.02.
- CLAP wins decisively — 3rd task (after FMA, UrbanSound8K) where CLAP tops fine-grained classification despite not having top coarse cohesion.
- audio_jepa near-bottom despite bird calls being well above its 20Hz cutoff — argues *against* the frequency-cutoff explanation being general, *for* the coarse-cohesion-over-fine-separability account being dominant.

**2026-08-10 (cont'd) — BirdCLEF re-run at 9 models (adds panns_cnn14, bird_mae).**
- MLP/KNN: bird_mae **0.345**/0.183; panns_cnn14 0.308/0.134; clap 0.284/**0.189**; hubert 0.260/0.068; mert 0.250/0.068; musicfm 0.247/0.113; music2vec 0.121/0.049; audio_jepa 0.089/0.043; wav2vec2 0.049/0.035.
- Pre-registered prediction 2 (BirdMAE domain-relevance test): BirdMAE beats CLAP on MLP (0.345 vs 0.284) but CLAP still barely ahead on KNN (0.189 vs 0.183, ~3% relative, arguably noise).
- **Complication**: PANNs shows the identical MLP-wins/KNN-loses-to-CLAP pattern despite zero domain match to birds — argues against pure domain-relevance, for a training-objective account: discriminative/contrastive (CLAP) → high raw separability (KNN); supervised classification (PANNs) or reconstruction (BirdMAE) → higher trainable-head exploitability (MLP) but weaker raw separability.

**2026-08-10 (cont'd) — BirdCLEF at 11 models (adds ast, audiomae).**
- MLP/KNN: **ast 0.394/0.239** (clean sweep, both best); bird_mae 0.345/0.183; panns_cnn14 0.308/0.134; clap 0.284/0.189; hubert 0.260/0.068; mert 0.250/0.068; musicfm 0.247/0.113; audiomae 0.167/0.039; music2vec 0.121/0.049; audio_jepa 0.089/0.043; wav2vec2 0.049/0.035.
- AST refutes "non-contrastive trades KNN for MLP": AST is supervised, AudioSet-trained, and beats CLAP's KNN outright.
- AudioMAE vs Bird-MAE (same objective, only domain differs): Bird-MAE MLP 0.345 vs AudioMAE 0.167 (>2x); KNN 0.183 vs 0.039 (>4x) — cleanest isolated domain-relevance test in the project, domain relevance is real and additive (not the sole explanation — PANNs already showed the same MLP/KNN split without domain match).
- Revised account: discriminative training pressure (contrastive OR supervised-with-labels) → high KNN AND high MLP simultaneously; reconstruction training → weaker KNN, still-reasonable MLP; architecture (PANNs' CNN) may modulate conversion efficiency without being disqualifying.

**2026-08-16 (part of Stage 4 gap-fill, dated 2026-08-16 entry references) — 18-model X-ARES coverage complete** for FMA-genre/UrbanSound8K/LibriSpeech-100h/BirdCLEF (previously uneven: 7/18, 11/18). Params/RSA scaling re-run clean: log(params) vs accuracy rho=-0.100 (p=0.69); discriminative-objective vs accuracy residuals rho=0.187 (p=0.46) — both non-significant at n=18. RSA-vs-params (direct Huh et al. 2024 scale-convergence test) rho=-0.100, p=0.70 — flat null, first direct test of this claim in the project.
- Whisper (smallest model, 20.6M encoder params) scores among highest accuracy (0.57) — visual outlier, doesn't move aggregate correlation.

**MIMII X-ARES probe (leave-one-physical-unit-out, 16 folds, private task).** 13/18 models with valid results at time of logging (2026-08-16/17): bird_mae 0.785/0.684; clap 0.730/0.731; data2vec_audio 0.826/0.825; hubert 0.836/0.620; musicfm 0.816/0.673; wav2vec2 0.824/0.725; audiomae 0.754/0.664; mms 0.846/0.792; panns_cnn14 0.812/0.672; sew 0.845/0.777; unispeech_sat 0.815/0.730; wav2vec2_conformer 0.788/0.703; wavlm 0.822/0.570 (MLP/KNN). Class imbalance ~4.5:1 normal:abnormal stated explicitly.

**ShipsEar 3-way task (A/B/D classes only, leave-one-session-out, 11 models).** All above ~33% chance floor (0.39–0.49 MLP) except **wav2vec2 at 0.127 — below chance**. Investigated: both wrappers produce healthy non-degenerate features on synthetic input (no NaN/zero); wav2vec2's raw feature dynamic range wider (min -3.12 vs hubert's -1.44) — read as "the probe failed to train stably for wav2vec2 here," not "wav2vec2's representation is uniquely bad." Not root-caused further (e.g. normalization before the probe untested).

---

## 3. Stage 5 — OOD fine-tuning (DeepShip, MIMII, confidential vessel, LoRA/ALLoRA)

**2026-08-10 — DeepShip metafile integrity failure.** `record_id` unreliable as a join key at scale: tanker record `47` maps to two different vessels/durations in the metafile, and neither matches the hosted file's actual measured duration (21s). Systematic check of all 63 files: Cargo 10/12 mismatched, Passengership 2/20, Tanker 15/28 (13 more only "resolved" by nearest-duration guess), Tug 0/3 (n=3, uninformative). **Consequence**: vessel identity/date/session cannot be recovered reliably — blocks vessel-grouped split. User-approved fallback: file-level grouping (63 files as 63 groups), 3 folds (fold0 183 / fold1 182 / fold2 180 clips at 10s each).
- Frequency-domain preservation audit across 7 models: only `audio_jepa` has an explicit low-freq cutoff (kaldi.fbank `low_freq=20`, from the upstream repo, not introduced locally) — all other 6 models (hubert/wav2vec2/mert/music2vec: raw-waveform CNN, no cutoff possible; clap: frequency_min=0; musicfm: torchaudio MelSpectrogram f_min=0.0) have none.

**2026-08-11 — Stage 5 v1: matched LoRA fine-tuning on DeepShip, 5 models (wav2vec2/hubert/mert/music2vec/ast, q_proj/v_proj, rank=8/alpha=16/dropout=0.05), 3-fold × 10 epochs.**
- Per-fold accuracy: music2vec 0.258/0.350/0.576 (mean 0.395); ast 0.249/0.283/0.611 (0.381); mert 0.204/0.272/0.653 (0.376); hubert 0.330/0.317/0.389 (0.345); wav2vec2 0.367/0.161/0.389 (0.306).
- Frozen MLP vs LoRA-adapted delta: mert 0.494→0.376 (**-0.118**); wav2vec2 0.415→0.306 (**-0.109**); music2vec 0.408→0.395 (-0.013); hubert 0.364→0.345 (-0.019). All 4 negative. Spearman(frozen, adapted) = 0.000 (p=1.000, n=4).
- Corroborating signal: wav2vec2's 1-epoch smoke test (0.511, fold 0) beat its 10-epoch result on the same fold (0.367) — more training made it worse.
- Reading: overfitting on tiny data (63 clips, ~40/fold train, tug class only 3 vessels total). Breadth-cluster correlation deliberately **not computed** — would be "noise regressed on noise" with n=5 across 3 groups (2/2/1).

**2026-08-16 — sample-rate corruption bug found in `stage5_lora_finetune.py` (DeepShip).** Passes model's *expected* rate into `resample()` as if it were the waveform's true native rate — silent no-op whenever they differ. DeepShip's clips are natively 32kHz vs every model's 16/24kHz expectation — v1's LoRA fine-tuning trained on ~1.3–2x-speed corrupted audio the entire run. Per user direction, DeepShip deprioritized — script left unfixed, v1 finding must be read with both the overfitting AND corruption confound stated, not as clean. (Confirmed the confidential-vessel scripts do NOT share this bug — they use per-clip true native rate throughout.)

**2026-08-16 — MIMII selected as 6th OOD domain** (valve/pump/fan/slide-rail, normal/anomalous), over InsectSet459 (messy sample-rate range 8–500kHz) and ICBHI 2017 (unclear license). 18,019 clips, 3 SNR tiers × 4 machine types; only 6dB tier (~32GB) downloaded. CC-BY-SA 4.0, verified via Zenodo API.
- Probe-set category `machine_sounds` added incrementally (12,000 rows / 6 categories total) without re-touching the existing 5.

**2026-08-16 — MIMII LoRA production run, 5 models, leave-one-machine-type-out, 4 folds.** Mean accuracy: ast 0.499, music2vec 0.489, mert 0.485, hubert 0.477, wav2vec2 0.458 (chance=0.50, binary balanced) — all at/near chance. Explicitly not comparable to the 16-fold leave-one-unit-out X-ARES MIMII probe (0.75–0.85 MLP there) — different fold scheme, different class balance.

**2026-08-16 — sample-rate bug found and fixed in MIMII's own LoRA script.** `run_fold()` passed model's expected rate as native rate (harmless for 16kHz-expecting models, real corruption for mert's 24kHz). Fixed with `MIMII_NATIVE_SAMPLE_RATE = 16000` (verified via `sf.info()`). Post-fix smoke test on mert: training loss 0.685→0.109 over 5 epochs (was barely moving before), confirming genuine learning — but test accuracy still near chance (0.515), so the near-chance MIMII result is a real generalization difficulty, not primarily the bug.

**2026-08-16 — MIMII extended to all 14 LoRA-compatible models + new frozen counterpart, same leave-one-machine-type-out split, all 19 active models.**
- Both conditions cluster near chance: frozen 0.44–0.57, LoRA 0.44–0.50.
- **11 of 14 models show LoRA underperforming frozen** (mean delta **-0.027**); only clap +0.059, hubert +0.016, wav2vec2_conformer +0.016 go the other way.
- 5 frozen-only models (no LoRA config): audio_jepa 0.532, musicfm 0.518, panns_cnn14 0.518, encodecmae 0.514, audiomae 0.499.
- **Paired statistical check (user-requested)**: model-mean-level Wilcoxon p=0.042, t-test p=0.036 (barely significant) — but fold-level sign test (n=54) is **30/54 negative, p=0.497** (coin flip). "Significance" driven by a few large-delta models (data2vec_audio -0.098, unispeech_sat -0.081), not a broad pattern. Logged as "statistically detectable but small and not deeply robust," not "verified."

**2026-08-16 — easier MIMII variant: leave-one-unit-out within same machine type, same 4-fold cost.**
- Both conditions show real above-chance signal: frozen 0.49–0.68, LoRA 0.44–0.70.
- "LoRA underperforms frozen" pattern **does not replicate**: mean delta shrinks to **-0.013**, only 9/14 negative, Wilcoxon p=0.326, t-test p=0.426.
- Some models now favor LoRA (wavlm +0.114, data2vec_audio +0.078, hubert +0.032); others still favor frozen (mert -0.109, bird_mae -0.067, whisper -0.065).
- Reading: harder split's borderline signal was at least partly a floor-effect artifact (both conditions near chance amplifies stochastic noise into a consistent-looking direction), not a robust property. Net: genuinely mixed, model-dependent.

**2026-08-16 — first RSA/CKA pass on confidential vessel domain (frozen, 19 models).** `music2vec` and `data2vec_audio` dramatically isolated (RSA 0.08–0.23 and 0.11–0.33; CKA independently confirms 0.03–0.22 for both) — far more isolated than on the public probe set. Visible cluster resembling breadth-hypothesis pattern: broad-training cluster (ast/panns_cnn14/bird_mae/audiomae/clap/audio_jepa/whisper/musicfm, mutual RSA 0.6–0.9) vs narrow-speech wav2vec2-family cluster (hubert/mms/sew/unispeech_sat/wav2vec2/wav2vec2_conformer, mutual RSA 0.5–0.8). Not yet formally tested against the breadth framework on this domain.

**2026-08-17/18 — matched LoRA/ALLoRA-vs-frozen completed on FMA-genre/UrbanSound8K/BirdCLEF (all 14 LoRA models + ALLoRA), computed 2026-08-18.**
- Across 42 LoRA-vs-frozen and 41 ALLoRA-vs-frozen comparisons: **zero underperform frozen.** Mean delta +0.153 for both. By category: FMA-genre +0.121/+0.113, UrbanSound8K +0.205/+0.214, BirdCLEF +0.135/+0.130.
- Largest single gain: Whisper on UrbanSound8K, frozen 0.281 → LoRA 0.778 (**+0.497**).
- Direct opposite of MIMII's "LoRA underperforms frozen" pattern. Read: that pattern is specific to near-chance tasks (MIMII), not a general LoRA/ALLoRA weakness — small stochastic differences get amplified into a consistent-looking direction at the floor. MIMII's own numbers restated as still accurate (-0.027/11-of-14 harder split, -0.013/9-of-14 easier split, both matching exactly), just should not have been read as evidence about LoRA generally.

**2026-08-17/18 — ALLoRA method adopted** (Huang & Balestriero, arXiv:2410.09692) after a research agent checked 7 candidate LoRA alternatives against their actual papers; most target *too little capacity* (memorization/backdoor problems), which doesn't match this project's own evidence (LoRA underperforming frozen probing, 1-epoch beating 10-epoch on the same fold — reads as an overfitting/optimization failure, not a capacity ceiling). ALLoRA targets exactly that regime (limited data, short runs), needs zero architecture change. Implemented as a custom `torch.autograd.Function` (removes LoRA's dropout/alpha scaling; rescales gradient into A/B per output-row by 1/sqrt(||(BA)_i,:||+1/eta²), leaves grad_input untouched).
- Validated: grad_input passed `torch.autograd.gradcheck`; grad_A/grad_B checked against an independent hand-written loop-based reference (not gradcheck, since ALLoRA deliberately isn't the true gradient) — passed.
- 3 real bugs caught during 14-model smoke test: base model left fully trainable (forgot freeze — caught via 72M–568M "trainable_params" vs LoRA's ~300K–800K); wavlm expects plain `nn.Linear` attrs (added passthrough properties); whisper's `hasattr(model, "base_model")` peft-detection collided with a real HF-native property (fixed to check `peft_config` instead).

**2026-08-18 (continued 3) — the actual Stage 5/6 ceiling correlation: does frozen geometry predict OOD adaptability?** (TwoNN ID, uniformity, alignment vs. LoRA gain, per model.)
- **MIMII: clean null.** All |rho| ≤ 0.30, p ≥ 0.30 across TwoNN/uniformity/alignment vs. both delta and raw accuracy. Survives the sample-rate, floor-effect, and overfitting confounds already separately identified. n=14 power caveat stated explicitly.
- **FMA-genre/UrbanSound8K/BirdCLEF: real positive result for uniformity/alignment (not TwoNN).**

  | Category | uniformity ρ | p | alignment ρ | p | TwoNN ρ | p |
  |---|---|---|---|---|---|---|
  | FMA-genre | +0.534 | 0.049 | -0.499 | 0.069 | -0.134 | 0.648 |
  | UrbanSound8K | **+0.798** | **0.001** | **-0.776** | **0.001** | -0.257 | 0.375 |
  | BirdCLEF | +0.116 | 0.692 | -0.156 | 0.594 | -0.033 | 0.911 |
  | Pooled | **+0.736** | **0.003** | **-0.732** | **0.003** | -0.231 | 0.427 |

  Direction sensible: AST/CLAP (uniformity ≈ -2.45, already spread) gain least (+0.07 to +0.09); wav2vec2/MMS/Whisper (uniformity ≈ -0.08 to -0.18, more collapsed) gain most (+0.18 to +0.31, Whisper highest).
- **Four follow-up checks, all landing the way needed for trust:**
  1. Uniformity vs. adapted (post-LoRA) accuracy directly: **null** (pooled rho=-0.319, p=0.267; all categories p=0.15–0.37). Precise wording: uniformity predicts *how much a model improves*, not *which model ends up best*.
  2. BirdCLEF null re-tested excluding bird_mae: rho moves from +0.116 (n=14) to **-0.016** (n=13) — collapses toward zero, hypothesis (domain-relevance swamping) did not hold.
  3. Parameter-count confound ruled out: n_params vs uniformity rho=-0.093 (p=0.752); n_params vs pooled gain rho=-0.272 (p=0.347); partial correlation uniformity-vs-gain controlling log(n_params) = rho=+0.754, p=0.002 (essentially unchanged from raw +0.736).
  4. Leave-one-model-out predictive validation: LOO MAE=**0.048** vs naive-mean baseline **0.060** (~20% real error reduction, n=14). Whisper worst-predicted (predicted +0.18, actual +0.31), named explicitly, not smoothed away.
- **Categorical-vs-continuous check**: objective-type discriminative vs. reconstruction — t-test p=0.078 (borderline), Mann-Whitney p=0.291 (not significant), disagreement because Whisper (discriminative) sits at uniformity -0.183, nothing like AST/CLAP's -2.45. Breadth-cluster ANOVA (3 non-singleton groups) F=9.11, **p=0.0087**, but Kruskal-Wallis across all 6 groups p=0.173, collapsed narrow-vs-broad p=0.347. Verdict: neither clean binary — uniformity isn't fully independent of category (AST/CLAP drive both), but the coarse categorical variables *fail* to predict gain where continuous uniformity succeeds (see C/D below) — reported as nuance, not collapsed to one verdict.
- **C: objective-type as categorical predictor of gain — null on both regimes.** MIMII: discriminative mean +0.0002 vs reconstruction -0.034, Mann-Whitney p=0.291. Pooled FMA/UrbanSound8K/BirdCLEF: discriminative +0.156 vs reconstruction +0.153, p=0.885.
- **D: breadth-cluster as categorical predictor of gain — null on both regimes.** MIMII: Kruskal-Wallis H=1.12, p=0.952 (collapsed narrow-vs-broad p=0.635). Pooled: H=9.00, p=0.109 (collapsed p=0.733).
- **Net Finding 5, worded precisely**: geometry does not predict OOD adaptability in general (MIMII null stands). On tasks with real signal (not BirdCLEF), uniformity specifically predicts adaptation *headroom* (not final adapted quality), confound-checked, out-of-sample validated. Coarse categorical variables (breadth, objective-type) that explain frozen-RSA structure and in-domain skill do NOT explain this. Do not simplify to "geometry predicts OOD performance."

**2026-08-18 (continued 4) — confidential vessel-domain version of the ceiling correlation.** All three metrics null: uniformity rho=+0.187 (p=0.523), alignment rho=-0.152 (p=0.605), TwoNN rho=+0.451 (p=0.106). Vessel gain small/mixed: mean +0.028, 3/14 models negative; accuracy narrow band (frozen 0.16–0.23, lora 0.15–0.32) — MIMII shape, not FMA/UrbanSound8K shape. **Matches the pre-registered prediction stated before the run finished.** Closes Finding 5's OOD story across all three domains: MIMII null, vessel null, FMA/UrbanSound8K/BirdCLEF positive-in-2-of-3.
- Real gap found+fixed same session: neither `rsa_cka_vessel.py` nor `run_all_vessel_experiments.py` had an ALLoRA condition (both predate ALLoRA adoption) — added, not yet run at time of that entry.

**2026-08-19 — vessel-domain ALLoRA run completed** (both accuracy and RSA/CKA geometry, on TCA). One CUDA OOM (wav2vec2_conformer, concurrent-job GPU contention) — checkpoint/resume worked, only 3/14 models needed rerunning.
- ALLoRA tracks LoRA closely: mean d_allora=+0.0273 vs LoRA's +0.0282, both 3/14 negative — same near-chance/floor-effect shape as MIMII-style tasks.
- Geometry-vs-adaptability null replicates under ALLoRA specifically: uniformity rho=+0.134 (p=0.648), alignment rho=-0.169 (p=0.563) — confirms the vessel null isn't a LoRA-specific artifact.

**2026-08-18 (continued 6) — LibriSpeech speaker-ID prediction, pre-registered before `finetune_librispeech.py` written.** BirdCLEF (50-way) was the one real-signal task where the uniformity-headroom effect went null while coarser FMA-genre (8-way)/UrbanSound8K (10-way) showed it. LibriSpeech speaker-ID (251-way, `train.clean.100`, 28,539 utterances, speaker_id field verified via HF datasets-server schema) is even finer-grained. **Prediction**: if task *granularity* (not something BirdCLEF/bioacoustic-specific) drives the null, LibriSpeech speaker-ID should also come back null/weak. Either outcome extends Finding 5.
- **Manifest built 2026-08-19**: 10,026 clips, 251 speakers (all kept, none dropped below the 20-utterance floor), train=7018/val=1503/test=1505, per-speaker-stratified split (utterance-level disjoint, full speaker overlap by design — closed-set classification, not a held-out-entity split).
- **Fine-tuning batch (47 jobs: 14 lora + 14 allora + 19 frozen) submitted and 45/47 complete as of 2026-08-19.** Only `wav2vec2_conformer` (lora+allora) outstanding — failed 3 times in sequence: MIG-slice OOM (transient contention, matched an identical failure on a different job), full-80GB-H100 OOM with a *different* signature (at model-*loading* time via `.to(device)`, not during training — this exact model already succeeded on FMA-genre/UrbanSound8K/BirdCLEF/vessel this session, so read as node-level contention on a specific bad node (`fc10512`) rather than a persistent bug). Retried a third time as of the last entry; not yet confirmed landed. **Correlation against Finding 5 (does 251-way replicate BirdCLEF's null) not yet computed — blocked on this one model.**

---

## 3.5 ICLR-submission robustness pass (2026-08-19) — framing rewrite, multiple-comparison correction, and stress-tests on Finding 5

Prompted by an external review of CLAUDE.md/the write-up identifying real gaps before submission-readiness (~3 weeks to deadline at the time). Five ordered items; items 2-4 run together per explicit direction once item 1's result narrowed the headline claim.

**Item 0 — central-contribution framing rewrite.** CLAUDE.md's title/one-line-summary/"Why this project exists" rewrote from the original (never-substantively-pursued) consensus-RDM/JEPA-relational-distillation Phase 2 framing to the actual 5-finding thesis. Added an explicit "Note on scope" marking the original plan as an abandoned early direction — Phase 1 reached decision-rule outcome 2 early (2026-08-09) and technically licensed proceeding to Phase 2, but the more informative direction that actually emerged was characterizing what explains cross-model agreement and whether it predicts anything functional, not building a distillation pipeline on a partial-agreement signal. Preserved as documented history, not deleted, per the project's own standing convention. "ECHO" adopted as the project name throughout (never actually appeared in-repo before this pass; "AudioRepBench" was external-discussion-only).

**Item 1 — Benjamini-Hochberg FDR correction across every reported p-value.** Compiled every p-value backing the *current/final* version of each of the 5 findings (interim/superseded roster-size checks excluded — e.g. Finding 1's 9/11/12-model permutation tests, already re-run at n=18; brain_rsa excluded per its own separate-track status). **Initial compilation: 51 values, 10 survive at alpha=0.05.**
- Survives: both Finding-1 permutation gaps (n=18); Finding 4's uniformity-KNN in both scopes (uniformity-MLP does *not* survive in either — **this sharpens Finding 4's own directional claim rather than weakening it**); Finding 5's pooled uniformity/alignment-vs-gain and UrbanSound8K's individual correlation; the parameter-count-confound partial correlation; the breadth-ANOVA-on-uniformity result (survives by rank, but its already-documented fragility — driven by AST/CLAP specifically, evaporates under more conservative tests — stays attached regardless).
- **Does not survive: Finding 5's FMA-genre individual per-category correlation** (raw p=0.049 → q=0.208) — a real downgrade. The precise post-correction framing: "UrbanSound8K individually significant, FMA-genre directionally consistent but not independently significant, pooled effect robust" — not "2 of 3 categories."
- **Interpretive caveat added explicitly** (per direct instruction, to prevent the ratio being misread): most of the ~41 non-survivors were never independent discovery claims — confound checks, falsification tests, and already-reported nulls whose entire point was to come back non-significant. "10 independently-significant claims survive, out of a full accounting including every confound check and null ever run" is the accurate reading.

**Items 2+4 — leave-one-family-out and clip-level bootstrap, run together** (item 4 elevated to equal priority with item 2 once item 1 narrowed the claim to the pooled effect + UrbanSound8K).
- **Item 2**: dropping wav2vec2-lineage (7 of 14 models: wav2vec2/hubert/wavlm/mms/unispeech_sat/sew/wav2vec2_conformer) leaves rho=+0.679 (p=0.094, n=7) — direction/magnitude hold, significance lost to reduced power, not a changed effect size. Dropping data2vec-lineage (2 models: data2vec_audio/music2vec) leaves rho=+0.755 (p=0.0045, n=12) — essentially unchanged, remains significant. **Not carried by any single family.**
- **Item 4**: real data-availability caveat stated up front — per-clip test predictions from fine-tuning were never saved, so only the *uniformity* side of the correlation could be clip-bootstrapped, not the *gain* side (partial, not full, robustness check). 300 bootstrap resamples of the shared 16,000-clip pool gave a **bit-identical rho=+0.736 across every resample** (std~1e-16) — verified real, not a bug, by directly confirming per-model uniformity does vary across resamples (e.g. `ast`: -2.34 to -2.40 across 5 draws, real ~0.06 spread) but this is tiny next to the ~2+ unit between-model gap (AST/CLAP ~-2.45 vs. wav2vec2/MMS ~-0.08 to -0.18), so no resample ever reorders the 14 models and Spearman rho (rank-based) never moves. Read precisely: rank ordering is extremely robust to clip-sampling, not "zero uncertainty in any absolute sense."

**Item 3 — baseline comparison table**, same pooled-gain target, 14 models: uniformity (rho=+0.736, p=0.0027) dramatically outperforms every real alternative tried — TwoNN intrinsic dimension (-0.231, p=0.427), log(n_params) (-0.272, p=0.347), the original Stage 1 silhouette/cohesion proxy (-0.218, p=0.455), and average RSA to the rest of the roster (-0.301, p=0.296) all show no signal at all. Naive mean-predictor baseline: MAE=0.0595 vs. uniformity's already-established real-model LOO MAE=0.048.

**Item 5, AMI/SingVERSE half — noise-robustness prediction tested, not confirmed, genuinely mixed.** Operationalized as within-category-normalized cross-domain distance (`speech` vs. `speech_noisy`, `music` vs. `music_noisy`; `results/noise_robustness_shift.csv`, 19 models). `speech_shift`: direction matches the prediction (broad-exposed mean=1.542 < narrow-clean mean=2.095) but n=3-vs-2 is far too underpowered (Mann-Whitney p=0.400). `music_shift`: **opposite** direction (broad mean=1.514 > narrow mean=1.358, p=0.800). Neither survives FDR. A legitimate reportable negative result — resolves the noise-confound Known-Risks caveat without becoming a sixth finding, not hidden.

**FDR family updated to 55 values once items 2 and 5's new p-values landed** (folded into the same corrected table per explicit instruction, not reported as clean standalone numbers alongside a corrected table for everything else). **11 of 55 now survive** — the data2vec-lineage leave-one-family-out result (p=0.0045) joined the survivors; neither new noise-robustness p-value does. Full ranked table (both the 51-value and 55-value versions) in journal.md, 2026-08-19 entries.

**Item 5, LibriSpeech half**: see §3 above — blocked on `wav2vec2_conformer`, not yet computed.

---

## 4. Geometry / Stage 6 diagnostics (silhouette → alignment/uniformity)

**2026-08-10 — silhouette proxy** (see §2, 7-model result rho=0.357/p=0.432 vs X-ARES MLP) — superseded by real alignment/uniformity below.

**2026-08-10 — Wang & Isola alignment/uniformity, first computed (9 models, same-category-proxy for "positive pairs," not true instance-level pairs).**
- Alignment and uniformity **perfectly rank-correlated** (Spearman = -1.0, Pearson = -0.991, p<1e-6) — one axis in this data ("embedding-space dispersion"), not two independent ones. Stated as a real methodological caveat, not glossed over.
- FMA+UrbanSound8K (7 models): uniformity→MLP spearman **-0.964, p<0.001**; uniformity→KNN -0.750, p=0.052 (marginal).
- BirdCLEF (9 models): uniformity→KNN -0.633 (p=0.067) > uniformity→MLP -0.400 (p=0.286), neither significant — pattern *flips direction* vs. the 7-model scope.
- Verdict: genuinely mixed on the KNN-vs-MLP mechanistic hypothesis (true in one scope, false in the other, both underpowered n=7/n=9). Uniformity-predicts-MLP in the 7-model roster is solidly established (p<0.001) regardless.

**2026-08-10 (cont'd) — true instance-level positive pairs (fixing the degeneracy).** Built `build_augmented_probe_subset.py` (100/category, 500 total, +2 semitone pitch shift). Added `alignment_score_paired()` (literal Wang & Isola definition).
- Paired-alignment vs uniformity: Spearman **-0.917 (p=0.001)** — down from -1.000, real but partial decorrelation.
- BirdCLEF: paired_alignment vs KNN = **+0.717, p=0.030** (newly significant) vs MLP +0.583, p=0.099 (not significant) — matches the PANNs/BirdMAE mechanistic hypothesis direction.
- FMA+UrbanSound8K: paired_alignment vs MLP = **+0.821, p=0.023** (significant) vs KNN +0.536, p=0.215 (not significant) — same scope-dependent reversal survives the fix.
- Verdict: degeneracy confirmed fixable, but the core scope-disagreement is a real roster-composition effect, not purely an artifact of the degenerate proxy.

**2026-08-15 — alignment/uniformity re-run at n=17 (full X-ARES coverage), resolves the mixed result.**
- Both scopes now agree uniformity predicts KNN more strongly than MLP: BirdCLEF uniformity-KNN -0.679 (p=0.003) vs uniformity-MLP -0.583 (p=0.014); FMA+UrbanSound8K uniformity-KNN -0.679 (p=0.003) vs uniformity-MLP -0.471 (p=0.057, no longer significant).
- Read: earlier disagreement (n=7/n=9) was an underpowered fluke, resolved by more data in the mechanistically predicted direction — not a real scope disagreement after all.

---

## 5. brain_rsa (side detour, kept scoped separate from the main roadmap)

**2026-08-13 — setup.** Uses Tuckute, Feather, Boebinger & McDermott (2023, PLoS Biology) via their own `gretatuckute/auditory_brain_dnn` repo directly (imports their RSA/correlation-matrix functions, not reimplemented, to avoid methodology mismatch — except where `h5py` unavailability forced a verbatim-copy inlining, see below).

**v1 — final layer only, NH2015, whole-brain, 5 models (wav2vec2/hubert/wavlm/whisper/ast).**
| model | RSA (spearman, mean±sem, n=8 participants) | noise-corrected |
|---|---|---|
| wavlm | 0.399 ± 0.015 | 0.526 |
| hubert | 0.361 ± 0.018 | 0.475 |
| whisper | 0.298 ± 0.022 | 0.393 |
| wav2vec2 | 0.252 ± 0.018 | 0.332 |
| ast | 0.160 ± 0.018 | 0.211 |

Noise ceiling (leave-one-out) = 0.759 ± 0.012. All 5 models well below it — expected for final-layer-only (paper's headline is about middle layers, not last).

**v2 — per-layer, per-ROI (Primary/Lateral/Anterior/Posterior).** Noise ceilings: AllROI 0.759, Primary 0.669, Lateral 0.723, Anterior 0.486, Posterior 0.445.
- Qualitatively reproduces the paper's core claim: mean peak-layer depth (0=shallow,1=deep) Primary **0.167** vs non-primary **0.408**.
- Per-model: 4/5 (hubert/wav2vec2/wavlm/whisper) show clear primary-shallower/non-primary-deeper split (e.g. wav2vec2 layer 0/25 Primary vs ~12/25 non-primary; hubert layer 2/25 vs ~11/25).
- **AST is the exception** — peak layer roughly the same mid-depth (fraction ~0.42–0.50) across every ROI including Primary, no differentiation.
- Best overall AllROI predictor across layers: **wavlm, RSA 0.473, noise-corrected 0.624, at layer 3/13** (an early-ish layer, not the final one v1 implicitly assumed).
- Full table: `brain_rsa/results_per_layer_nh2015.csv`, 415 rows (5 models × ~16.6 layers avg × 5 ROI conditions).

**v3 — per-category (11 semantic categories from `cat_assignment`).** Counts: Mechanical 39, EnvSound 27, Music 24, HumNonVoc 15, HumVoc 13, Song 11, EngSpeech 10, AniVoc 10, ForSpeech 7, AniNonVoc 5, Nature 4 (sum 165).
- Confirms averaging hid opposite trends: wavlm+ast climb steadily with depth on Music (peak 60–100%); wavlm *peaks early* (~10%) then declines on Mechanical (largest category, n=39) — opposite-direction curves a pooled number would blend flat.
- EngSpeech: ast spikes sharply ~30–40% depth (RSA~0.57) — sharper speech-specific signature than ast shows anywhere else, despite ast not being speech-domain.
- Smallest categories (ForSpeech n=7, AniNonVoc n=5, Nature n=4) visibly noisy, sometimes RSA<0 — flagged as exploratory only.

**B2021 support added** (192 stimuli, confirmed matching NH2015's 165 in identical order). Per-category comparison against NH2015 not yet done.

**Model×model×brain intercorrelation heatmap (NH2015).** Models agree with each other (0.6–0.96) far more than any agrees with the brain (mostly 0.03–0.66). HumNonVoc/AniNonVoc: wav2vec2/hubert/wavlm nearly redundant (0.90–0.96), all peaking at 0% depth. Foreign Speech (n=7) worst panel: wavlm-whisper intercorrelation **negative** (-0.29). AST's peak-matching depth swings 8%–100% across categories vs. the other four's stable 0–33%.

**Depth-quartile × category.** The whole-dataset AST-decoupling-with-depth finding is **category-dependent, not a fixed property**: Mechanical (n=39) shows it sharply/monotonically (AST-hubert RSA 0.57 at 0–25% depth → -0.01 at 75–100%); Instr. Music (n=24) shows the same direction, much weaker/non-monotonic (0.46→0.29→0.41). The other 4 models stay tightly correlated (0.4–0.93) at all depths in both categories.

**2026-08-18 — mert + clap disambiguation (peer-relayed objective-vs-domain critique, verified before acting).** Δlayer_frac (Lateral peak − Primary peak), 7 models: wav2vec2 (masked) 0.33; hubert (masked) 0.80; wavlm (masked) 0.17; whisper (discriminative/ASR) 0.66; mert (masked, music) ≈0.08 flat; ast (discriminative, general) -0.08 flat; **clap (contrastive, general, audio-text) 0.25**, partial exception.
- Reading: domain-match (speech vs non-speech) predicts differentiation better than objective (masked vs discriminative) at n=7 — whisper is the clearest single counterexample to a pure-objective account. clap's partial differentiation flagged as possibly tied to natural-language supervision, a variable neither axis captures. Explicitly small-n/exploratory.
- Folded into `brain_rsa/proposal/proposal.tex` (Neurocomputing course-paper draft, 5-page body + 1 page refs) as motivating preliminary evidence.

---

## 6. Model-roster / checkpoint-provenance work

**2026-08-08 kickoff table corrections:** A-JEPA (Fei/Fan/Huang, original) has no public checkpoint anywhere → substituted `ltuncay/Audio-JEPA` (Tuncay et al., ICME 2025, MIT), labeled as substitute everywhere. Microsoft CLAP needs separate `msclap` pip package → dropped for `laion/larger_clap_general` (HF-native). BEATs/MusicFM need custom loaders (not plain `AutoModel`).

**2026-08-09 — music2vec correction (see §1).** data2vec-style: student encoder operates directly on masked input, predicts EMA teacher's averaged top-K layer reps, **no separate predictor network**. JEPA (per A-JEPA/Audio-JEPA's own methods sections) has 3 components: context encoder, EMA target encoder, and a **decoupled predictor network** — the actual dividing line. Corrected in CLAUDE.md's table/H1 (annotated in place, journal left as append-only record), `music2vec.py`'s `ModelInfo.paradigm`, README.
- New confound documented: `ltuncay/Audio-JEPA` trained on much less compute (100k steps, ~14h on 4 V100s, 5,338h AudioSet) vs wav2vec2/data2vec's 400k steps on larger batches; paper reports it substantially underperforming on linear probes (Speech Commands V1: 0.152 vs data2vec's 0.927).

**2026-08-09 (cont'd) — audio_jepa loader.** HF repo `ltuncay/Audio-JEPA` ships author `inference_example.py` + `config.json`; loads `VisionTransformer` standalone, restores only `encoder.*` keys (predictor+target_encoder discarded per the model's own config: "Only the encoder is used downstream"), `strict=True`, 0 missing/0 unexpected keys.
- `flash_attn` shim: repo's `Block` unconditionally imports compiled `flash_attn.modules.mha.MHA`; ported HF repo's own portable SDPA-based shim instead.
- Self-inflicted regression: `pip install flash_attn` silently downgraded torch 2.11.0→2.9.1 (ABI break for torchaudio/torchcodec/torchvision) — caught via an unrelated torchaudio import failure, fixed by uninstalling flash_attn + reinstalling torch==2.11.0 exactly.
- Short-clip bug: kaldi.fbank needs ≥~98ms input; some UrbanSound8K clips as short as ~60ms crashed the first full run (missed by a smoke test using only a 5s dummy). Fixed with a pad-to-10s floor.

**2026-08-10 — checkpoint_status schema formalized** (`audio_comp/models/base.py`): 4-value enum (`official_open_weights`, `official_public_weights_license_unclear`, `community_conversion`, `code_only`). Enforced at `register_model()` (valid value) and `get_model_class()` (comparison-eligible), the single chokepoint every pipeline entry point resolves through.
- `beats` license: re-verified rather than blindly applying the instructed label — unilm root LICENSE is MIT, BEATs subdirectory README defers with no carve-out. Landed on `official_open_weights` but documented as a judgment call (absence-of-carve-out, not explicit per-checkpoint citation), and confirmed this doesn't change beats' deferred status (loader-only gap, not a license blocker).

**2026-08-10 — PANNs CNN14** (`github.com/qiuqiangkong/audioset_tagging_cnn`, MIT, checkpoint on authors' own Zenodo 3987831). Used `Cnn14` architecture class directly, not the `AudioTagging` wrapper (hardcodes `$HOME` path, forces DataParallel). Smoke-tested clean (1s + 60ms). **Short-clip pooling-floor bug**: 5 stages of (2,2) time-pooling collapse the logmel time axis to zero for clips <~400ms (`RuntimeError: Given input size ... Calculated output size: (256x0x8)`); empirically tested threshold: fails up to 300ms, safe at 400ms. Missed by the original smoke test (60ms clip only tested padded inside a batch, never standalone at batch=1) — this exact blind spot recurred a 2nd time after audio_jepa's kaldi.fbank case, prompting the permanent "test shortest clip standalone, batch size 1" smoke-test rule.

**2026-08-10 — Bird-MAE** (`DBD-research-group`, arXiv 2504.12880, confirmed same group as BirdSet paper). **No LICENSE file in the GitHub repo at all** (GitHub license API returns 404, stronger absence than BEATs' case) and no HF card license field → `checkpoint_status="official_public_weights_license_unclear"`, still comparison-eligible but flagged as needing an author email before use beyond internal comparison.
- Version-skew bug: checkpoint's `config.json` declares transformers 4.38.0, venv has 5.15.0; `BirdMAEModel.__init__` never calls `self.post_init()` → `AttributeError: 'BirdMAEModel' object has no attribute 'all_tied_weights_keys'`. Fixed with a narrow monkeypatch on `PreTrainedModel._move_missing_keys_from_meta_to_device` (active only during `load()`), not a transformers downgrade.

**2026-08-10 — AST** (`github.com/YuanGongND/ast`, BSD-3-Clause). Uses the AudioSet-*trained* checkpoint (not "base non-finetuned") deliberately — AST has no two-stage self-supervised-then-finetuned structure, its supervised training *is* its pretraining.

**2026-08-10 — AudioMAE** (`facebookresearch/AudioMAE`, CC-BY-4.0, explicitly covers weights). Model-construction kwargs read directly out of the checkpoint's saved `args` Namespace, cross-checked against tensor shapes (`pos_embed` (1,513,768) ⇒ 512 patches + cls token). **4 separate version-skew fixes**, all narrow shims: (1) `torch._six.inf` removed in PyTorch 2.0 — patched; (2) old timm API needed for `Block(qk_scale=...)` — pinned `timm==0.4.9` project-wide, confirmed via empty `Required-by` that nothing else depends on timm; (3) same old-timm requirement conflicts with the checkpoint's Swin-based decoder needing a newer timm API — no single timm version has both (tested 0.3.2/0.4.9/0.4.12/1.0.28) — resolved by constructing with `decoder_mode=0` (decoder never used) and `strict=False` load with a runtime assertion every unmatched key is decoder-prefixed; (4) `np.float` removed in numpy≥1.24, restored as alias (harmless, value immediately overwritten by `load_state_dict`).

**2026-08-10 — Perch 2.0 checked, deferred.** Chen et al. arXiv:2508.04665 — objective is hybrid (supervised multi-taxa + self-distillation prototype-learning + "source-prediction"), doesn't cleanly fill the discriminative/bioacoustic cell. Also TensorFlow-only (Kaggle SavedModel, Apache-2.0, no PyTorch/JAX/ONNX).

**2026-08-10 — BirdNET added as isolated-environment exception, not in the `audio_comp.models` registry.** Kahl et al., `github.com/kahst/BirdNET-Analyzer` — clean supervised classification (sigmoid+BCE multi-label), better conceptual fit than Perch 2.0. Code MIT, **model weights CC BY-NC-SA 4.0** (more restrictive than most of the roster). TensorFlow/TFLite-only → fully separate CPU-only venv (`$SCRATCH/birdnet-venv`), standalone script writes `.npz` in the exact schema `extract_embeddings.py` produces. Not covered by `registry.py`'s enforcement (no `BaseAudioEncoder` subclass possible across a process boundary) — provenance documented in CLAUDE.md/script docstring instead. Own checkpoint-download URL 404'd (tuc.cloud, stale) → used the official Zenodo v2.4 TFLite release directly. Bypassed the higher-level `embeddings()` API (needs `perch_hoplite`, failed to resolve) for the lower-level `model.embeddings(sample)` TFLite call.

**2026-08-12 — Tier 2 additions (18 total active, batch):** WavLM (`microsoft/wavlm-base-plus`, CC BY-SA 3.0 Unported, verified against `microsoft/UniSpeech`'s root LICENSE, not assumed shared with BEATs' `microsoft/unilm`), Whisper (`openai/whisper-base`, Apache-2.0, encoder-only, never `.generate()`), Data2Vec-audio (`facebook/data2vec-audio-base`, Apache-2.0, base checkpoint not `-960h`), MMS (`facebook/mms-300m`, CC-BY-NC-4.0, most extreme breadth data point, 1400+ languages), UniSpeech-SAT (`microsoft/unispeech-sat-base`, same repo/license as WavLM), SEW (`asapp/sew-tiny-100k`, Apache-2.0), wav2vec2-Conformer (`facebook/wav2vec2-conformer-rel-pos-large`, Apache-2.0). All smoke-tested per the standing edge-case rule.
- Deferred, not skipped: SpeechT5 (base checkpoint under non-official `ajyy/` namespace; official `microsoft/` variants are ASR-fine-tuned); EnCodec (`facebook/encodec_24khz`, MIT, but `transformers.EncodecModel` only exposes discrete post-quantization codes, no documented continuous-latent path).
- Real infra bug: `$HOME` quota hit 98% (47G/48G) mid-batch — MMS download failed. Interactive smoke tests weren't setting `HF_HOME` (only batch sbatch jobs were). Fixed by exporting `HF_HOME=$SCRATCH/hf_cache` and deleting the redundant `~/.cache/huggingface` (4.8GB).

**2026-08-16 — EncodecMAE wired up** (`github.com/habla-liaa/encodecmae`, MIT), model name `ec-ec-large_st` (verified by catching `load_model()`'s own error listing valid names — the HF repo's usage snippet's `large-st` doesn't exist). Short-clip crash: chunking only processes windows >2048 samples; 60ms at 24kHz = 1,440 samples → empty chunk list. Fixed with zero-padding.

**2026-08-16 — BEATs loader written, blocked on manual step.** Vendors 3 files from `microsoft/unilm/beats` (BEATs.py/backbone.py/modules.py only — Tokenizers.py/quantizer.py confirmed tokenizer-training-only via direct import inspection). No fairseq/hydra dependency. Checkpoint identity double-checked ("BEATs_iter3+ (AS2M)" is the *Pre-Trained Model* column entry, not an AudioSet-fine-tuned variant, despite the name looking like it could be). **Blocked**: checkpoint host is a OneDrive personal-share link, confirmed 403 via direct `curl -IL` — no programmatic download path exists (first such case in this roster). Waiting on user browser download.

**19 registered / 18 active as of 2026-08-12's batch** (per CLAUDE.md). Not verified whether this count changed further by 2026-08-19 in the read journal entries — see Gaps.

---

## 7. Infrastructure / environment incidents

**2026-08-08 — tokenizers/hf_xet deadlock.** hubert/mert intermittently deadlocked in `from_pretrained()` (confirmed via `/proc/<pid>/status`: alive, blocked on `futex_do_wait`, near-zero CPU). Initially misattributed to generic cluster load (~7000 load average, real but a red herring) until the user pushed back and the live process was actually inspected. Fixed: `TOKENIZERS_PARALLELISM=false`, `HF_HUB_DISABLE_XET=1`, `RAYON_NUM_THREADS=1`.
- Separate, genuine issue: mert's 1.26GB checkpoint download flaky (dropped at ~10%). Fixed by pre-fetching on the login node with a retry loop before the GPU job.
- fir Slurm needs explicit GPU type+MIG slice: `--gres=gpu:nvidia_h100_80gb_hbm3_1g.10gb:1`.
- `skdim` PyPI package is actually named `scikit-dimension`. `pyarrow` needs `module load gcc arrow/x.y.z` loaded at both install AND runtime (PYTHONPATH injection, not site-packages) — this specific module-version pin recurred as a live issue again 2026-08-16 (see below, `datasets==2.19.0` downgrade needed on TCA for a related but distinct torchcodec issue).

**2026-08-10 — `$HOME` storage incident.** File reads failing with `Input/output error`; `diskusage_report` stuck at 97MiB; `git status` → `"Cannot send after transport endpoint shutdown"`. Methodically isolated (not assumed cluster-wide): untouched `eddyflow-venv` read fine at the same moment (rules out filesystem-wide outage); `pip install --force-reinstall` failed reading pip's own vendored files (rules out "just corrupted numpy"); brand-new venv's `ensurepip` failed writing a brand-new file (rules out "old corrupted files specifically" — proved active/general `$HOME`-write failure). Rebuilt venv at `$SCRATCH/audio-comp-venv`. One corrupted inode (`audio_comp/models/__init__.py`, literal `-?????????` under `ls -la`) recreated from known-good committed content, verified byte-identical via `git status`. **Net: not data loss, not caused by anything the project did** — transient cluster storage backend issue, correlated with (not proven to cause) extreme login-node load (~7000-7600, 100+ users), seen at least twice this project. Switched `audio_comp` off `pip install -e .` to `PYTHONPATH`-relative (its wheel-build step was less robust to the intermittent failure).

**2026-08-10 — DeepShip download incident, same signature.** `git clone` hit the identical EIO/transport-shutdown on `.git/logs/HEAD` ref-log append; `df -h $HOME` showed a corrupted-looking report (`-11P` used, `471%`) alongside a normal `diskusage_report`. Switched to zip download (no ref-log write) rather than git clone as the general workaround.

**2026-08-10 — background `nohup` process killed by login-node reaper at 12 models.** `compare_models.py` silently killed within ~2.5 minutes, twice, no error/crash on foreground retry — almost certainly the reaper catching a CPU/memory-heavy background process. Fixed properly by making it a real Slurm job (`compare_models.sbatch`, `def-spadon_cpu`, 8 cores/32GB/1.5hr) instead of continuing to fight the login node.

**2026-08-12 — laptop GPU OOM, confidential Stage 5 v2 local script** (no confidential detail). A synthetic single-batch dry run predicted the run would fit; it OOM'd. Two compounding causes: (1) OS/display reserves real VRAM a headless dry run never sees (card reports 6144 MiB total, only ~5730 MiB actually free at idle); (2) real multi-batch training ran ~19% higher peak memory than one isolated forward+backward+step suggested (allocator fragmentation). Fixed: periodic `torch.cuda.empty_cache()` (every 20 batches + end of epoch), memory cap 80%→75%, halved batch-size recommendations. Generalizable lesson: a clean synthetic dry run measures a floor, not a safe operating point.

**2026-08-13 — multi-login-node tunnel confusion.** `login1`/`login2`/`login3` confirmed to exist (`login4` doesn't resolve); a fresh SSH/Bash session can land on any of them, and the laptop's reverse-SSH tunnel only listens on whichever node its outbound connection actually reached. A session checked its own node's `ss -tlnp`, saw nothing, wrongly concluded the tunnel was down — it was alive on `login3` while that session was on `login1`. Fixed by checking all known login nodes (`ssh -o BatchMode=yes <node> 'ss -tlnp | grep <port>'`, works node-to-node since `authorized_keys` is shared) before declaring a tunnel dead. Port 2222 shows LISTEN on every login node — confirmed to be an unrelated other user's process, not this project's (motivated the original move to a randomized loopback-bound port).
- Same session: laptop had a real hard reboot mid-extraction-run (critical battery, unrelated to inference load) — required resuming after the tunnel came back on a different login node.

**2026-08-12 — real bug in confidential local training script found while inspecting a laptop OOM (see above), listed again here as its own distinct fix: periodic empty_cache + conservative memory cap + halved batch sizes.** (Cross-referenced, not duplicated in count.)

**2026-08-16 — MIMII download bandwidth throttle, same class as bigdata6 streaming.** A Slurm CPU job downloading ~32GB (4 zip files) was still on file 1 of 4 after 2.5 hours (~400KB/s). Diagnosed via a 15-second `curl` test directly on the login node (~2MB/s, ~5x faster) rather than guessing. Fixed by killing the Slurm job, deleting the partial download, running via `nohup ... & disown` on the login node directly, watched through the ~2.5-minute reaper window (survived — I/O-bound curl is lower-risk than CPU/memory-heavy processes). Completed in under an hour once off the compute node.

**2026-08-16 — `wavlm.npz` missing from the embeddings dir despite being part of already-documented 18/19-model comparisons.** Root cause not investigated; fixed pragmatically by re-running full extraction (job 54869990, 20 min). Flagged as a risk pattern: an unexplained missing artifact for a model nobody re-touches could go unnoticed indefinitely.

**2026-08-16 — `$HOME` filled to 100% of 48GB quota (2nd major storage incident).** Discovered via `mert`'s MIMII job failing on `OSError: Disk quota exceeded` during an HF checkpoint re-download. Root cause: `~/audio_comp_data` (24G) and `~/audio_comp_external` (4.1G) had been silently living in `$HOME` the entire project (env vars `AUDIO_COMP_DATA`/`AUDIO_COMP_EXTERNAL` default to `~/...` paths, never overridden in `.bashrc`) — a pre-existing Phase 1 misconfiguration, not introduced this session, just never hit the ceiling until now. Plus `~/.cache/huggingface` (4.8G). Fixed: rsync'd all three (verified byte-identical size match first) to `$SCRATCH`, set the env vars permanently in `.bashrc`, deleted originals only after explicit user confirmation. 28GB reclaimed, `$HOME` now 32% full. Flagged as the likely (not retroactively re-diagnosed) root cause of at least one earlier "transient GPU driver error" MIMII failure.

**2026-08-16 — cluster-side transient GPU driver failure on MIMII X-ARES jobs.** `"Failed to get device handle for GPU 0: Unknown Error"` → `torch.cuda.device_count()` returns 0 → `ZeroDivisionError` in X-ARES's own `run.py` (`i % num_gpus`). Confirmed non-systemic by comparing failed logs against 2 jobs on healthy nodes in the same batch that succeeded (`bird_mae`, `clap`). Round 1: 12/18 failed this way, resubmitted. Round 2: resubmitted 12 succeeded; 2 new failures (`ast`/`whisper` timed out at 3h — 18,019-clip MIMII task bigger than any prior public dataset); 3 hit the identical driver error again (`audio_jepa`/`mert`/`music2vec` — bad node luck, none had failed this way in round 1). Round 3: extended timeouts, landed clean.

**2026-08-16 — second divergent-venv discovery.** `/scratch/pdoshi/audio-comp-venv` (the venv sbatch jobs actually use) was missing `datasets`/`pyarrow` entirely; a second, older `~/audio-comp-venv` in `$HOME` had `datasets==5.0.1` — the project had silently had two divergent venvs, likely from an earlier partial migration. Standard `module load arrow` fix didn't expose pyarrow to this venv — genuine unresolved environment issue, not chased further since the immediate plot didn't end up needing it.
- Same debugging pass: `AUDIO_COMP_DATA` was set in `.bashrc`, but the actual code reads `AUDIO_COMP_DATA_ROOT` — a different name, so the earlier fix silently never applied. Also discovered `sbatch --wrap` jobs do **not** source `.bashrc` at all (confirmed empirically) — every relevant sbatch submission needs `AUDIO_COMP_DATA_ROOT`/`AUDIO_COMP_EXTERNAL`/`HF_HOME` exported explicitly in the `--wrap` command itself.

**2026-08-16 — TCA (tca-s01.research.cs.dal.ca) brought online.** Shared 2× RTX PRO 6000 Blackwell Max-Q, plain SSH, no scheduler. Fresh venv, version-matched to fir (`torch==2.11.0+cu128`, `transformers==5.15.0`, `peft==0.20.0`, `huggingface_hub==1.27.0`). `data/AIS/` fully gitignored — an SFTP transfer of 3 clean `.py` files was blocked by the safety classifier for reaching into that directory even though the files themselves were clean; user had to `scp` them manually.
- Second env-skew round: `audiomae` crashed with `Block.__init__() got an unexpected keyword argument 'qk_scale'` — TCA's `pip install timm` pulled latest `1.0.28` vs fir's pinned `0.4.9`. Fixed by pinning on TCA too. Proactively load-tested all remaining 16 models on TCA rather than waiting to hit each serially — all loaded clean.
- `setup_audiomae.sh`'s gdown step invoked system Python (no gdown installed) instead of the venv's — fixed to call through the venv explicitly.
- Installing `timm` pulled a CPU-only `torchvision` as a side effect, silently incompatible with `torch==2.11.0+cu128` — `encodecmae`'s import chain hit `RuntimeError: operator torchvision::nms does not exist`; a plain `pip install torchvision` no-op'd ("already satisfied", pip didn't recognize the ABI mismatch) — fixed with `--force-reinstall --no-deps` against the cu128 wheel index.

**2026-08-16 — real password-handling incident.** User shared a reusable CSID password for shared multi-user TCA server, asking the assistant to use it directly. Flagged as a different risk class than a scoped API key (shared infrastructure, other users' running jobs at stake) — recommended the user drive login themselves. User explicitly confirmed scope ("can only use it for this thing") and asked the assistant to proceed anyway; complied but login failed with `AuthenticationException` (network reachable, password rejected); did not retry blindly to avoid account lockout risk — left for the user to resolve.

**2026-08-16/17 — TCA network-throughput anomaly.** A `salloc` (2g.20gb slice) job on `fc11020` burned its entire 3-hour allocation reaching only 800/2463 clips during prefetch (~180x slower than normal for this code path). Basic TCP connectivity to bigdata6 tested fast (~0.1s/connect) from a different node (`fc30669`) — rules out "compute nodes can't reach bigdata6" categorically. Couldn't test actual SFTP throughput without the password — root cause (bad node vs. general SFTP overhead vs. transient congestion) left unresolved; recommended a fresh `salloc` + quick-abort-if-still-slow rather than risk another 3-hour window blind.

**2026-08-17 — thermal shutdown on laptop (distinct from the earlier memory-only OOM).** Fan spun up hard then forced power-off, confirmed AC-powered (not battery). Memory fixes did nothing (thermal ≠ memory problem) — Max-Q mobile GPU has tight sustained-load thermal headroom. Added active temperature monitoring (checked every 5 batches): paces with short sleeps at 78°C, fully pauses (polls until <75°C) at 85°C — conservative consumer-GPU defaults, override via env vars. Flagged to user: software pacing can't fix genuinely inadequate physical cooling.
- Same day: tightened thresholds (65/78/60°C) still cycled continuously (climbed straight back to 86°C within moments of resume) — added consecutive-cycle escalation (3 cooldowns in a row → forced 90s rest). Even the lightest workload (frozen-embedding extraction, no backward pass) still cycled continuously at these thresholds — strong evidence physical cooling, not workload, is the bottleneck. User asked to loosen for speed; declined to remove the check but raised thresholds to 88/93/85°C at the user's informed request — still below the ~100–105°C typical hard-shutdown range.

**2026-08-17 — real bugs found post-batch (FMA-genre/UrbanSound8K/BirdCLEF LoRA/ALLoRA/frozen, 169 jobs).**
1. **OOM on a full 80GB GPU** for several large wav2vec2-family models on FMA-genre specifically — `generic_lora_trainer.py`'s loop never called `empty_cache()` (unlike every other fine-tuning script), compounded by FMA-genre's 30s clips being far longer than other tasks' 4–10s. Fixed with periodic `empty_cache()` + `max_duration_s` truncation (matched X-ARES's own `crop_length=10`).
2. **UrbanSound8K sample-rate bug, within a single dataset.** `finetune_urbansound8k.py` assumed uniform `NATIVE_SAMPLE_RATE=44100` for all clips (docstring itself flagged this as unverified). Checked directly: a 200-clip sample showed genuinely mixed native rates (44100/48000/96000/24000/16000/8000/192000) — **all prior UrbanSound8K results (frozen AND lora AND allora) silently corrupted for ~37% of clips.** Fixed: `read_mono()` returns true native rate per file, `read_batch_skip_bad()` resamples explicitly using it. FMA-genre and BirdCLEF independently verified genuinely uniform-rate (FMA 44.1kHz, BirdCLEF 32kHz, both checked via 200-clip sample, not assumed) — unaffected. Same `expected_sample_rate`-as-native bug pattern also found inside `train_and_eval_frozen()`'s `embed()` — fixed with an explicit required `native_sample_rate` parameter.

**2026-08-18 — fix-not-committed incident.** The UrbanSound8K sample-rate fix existed only locally for a while; TCA's `git pull` had nothing new to fetch, so it kept running the stale pre-fix trainer, reproducing the identical corruption there. Committed and pushed (`e87a1db`). Lesson: a fix isn't real until it's committed and reachable via `git pull` elsewhere.

**2026-08-18 — separate root cause of TCA's continued failures even after the git push landed.** TCA's `data/urbansound8k_manifest.csv` had every path corrupted with a literal unresolved placeholder string (`<TCA_UID>` never substituted with the real username) from an earlier `sed` fix run against the display placeholder instead of the resolved value. Fixed in place on TCA only (not committed — TCA/fir intentionally have different path prefixes in this file). Verified all 8732 paths resolve afterward.
- Per user direction, UrbanSound8K work moved off TCA entirely onto fir's Slurm queue; TCA reserved for vessel-data work only going forward. Checked `sacct` directly: most of the post-fix 47-job batch (55241500-55241620) had already completed successfully; only 6 hit time limits, resubmitted with more headroom.

**2026-08-18 — SingVERSE loader couldn't complete full shuffled run on fir's login node.** Killed once by the node's resource governor (SIGKILL, buffer_size=2000); timed out once even at buffer_size=50, purely from parquet shards being slow to fetch over the network (a single unshuffled row fetch alone took ~207s). Not a code defect — same proven wrapping pattern as `ami.py`, field access independently confirmed twice. Flagged: expect slow first-shard latency if this source is used in `build_probe_set.py` later.

---

## Not found / gaps in this inventory

**Resolved since the last pass (2026-08-19), removed from the open list:**
- ~~`results/finetune_librispeech_speaker.csv` result~~ — 45/47 jobs complete; the correlation-against-Finding-5 computation itself is what's still blocked (see below, now a narrower gap).
- ~~AMI/SingVERSE noise-robustness result~~ — tested, not confirmed, logged in §3.5.
- ~~`rsa_cka_vessel.py` LoRA-condition (post-LoRA-adaptation geometry) run~~ — confirmed complete via direct file check this pass (`vessel_rsa_matrix_lora.csv`/`vessel_cka_matrix_lora.csv` both exist, dated 2026-08-18), alongside the frozen and ALLoRA conditions. All three vessel geometry conditions done.

**Still open:**
- **`wav2vec2_conformer` LibriSpeech lora/allora, and the resulting LibriSpeech-vs-Finding-5 granularity correlation** — narrower than the original gap (45/47 other models done), but this specific model has now failed 3 times (MIG OOM, full-H100 OOM with a different signature, third retry not yet confirmed as of the last entry) and the correlation itself hasn't been computed pending it.
- **Registered/active model count as of 2026-08-19** — CLAUDE.md states "19 registered, 18 active" as of the 2026-08-12 batch; whether BEATs or any other model changed status between 2026-08-13 and 2026-08-19 is not stated in the journal entries read.
- **B2021 vs NH2015 per-category comparison** (brain_rsa) — flagged in the 2026-08-13 entry as "not yet compared," no later entry closes this.
- **Depth-quartile-by-category full cross-file summary** (brain_rsa, 11 per-category PNGs) — flagged as "not yet summarized across all 11 files," no later entry closes this.
- **BEATs checkpoint** — still blocked on the user's manual OneDrive browser download as of the last entry mentioning it (2026-08-16); no later entry confirms it was provided or that BEATs moved from `deferred_models` to `active_models`.
- **TCA SFTP throughput root cause** (§7, 2026-08-16/17 entry) — explicitly left unresolved ("bad specific node vs. general SFTP overhead vs. transient congestion... still unresolved").
- **wav2vec2's exact-zero LibriSpeech-ASR score and its general X-ARES weakness** — repeatedly flagged as "checked, not resolved" across multiple entries (2026-08-10 X-ARES, BirdCLEF, ShipsEar) but never root-caused to a specific mechanism.
- **`wav2vec2_conformer`'s repeated OOM pattern across both the vessel-ALLoRA-geometry run and the LibriSpeech run** — both attributed to node-level contention rather than a persistent bug (the model succeeds elsewhere: FMA-genre, UrbanSound8K, BirdCLEF, vessel accuracy conditions), but this is now the *second* distinct pipeline where this specific model has needed a retry-on-contention workaround. Worth a passing mention if it happens a third time somewhere new — currently read as coincidence, not yet enough evidence for a model-specific pattern.
