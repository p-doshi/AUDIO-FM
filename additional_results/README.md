# Additional results (not in the paper)

## mls_crosslingual/
Cross-model RSA on Multilingual LibriSpeech speaker clips (German 4,000, French 3,599, Italian 2,261, Dutch 1,272, Polish 355 clips;
19 models; frozen embeddings; RDM = correlation distance, RSA = Spearman between RDM upper triangles).

- `<language>_rsa_matrix.csv`: the 19×19 RSA matrix per language.
- `mls_summary_with_BH.csv`: tests per language, with Benjamini–Hochberg q computed within each family:
  - **Agreement with the English (LibriSpeech) RSA structure:** Spearman ρ = 0.54–0.60, permutation p ≤ 0.003. All 5 of 5 survive BH.
  - **Taxonomy alignment (Mantel r, domain+paradigm and breadth groupings, as in Finding 1):** 10 tests, smallest p = 0.018.
    **None of the 10 survives BH correction.** Treat the per-language taxonomy pattern as descriptive only.
  - **Uniformity rank agreement with English:** ρ = 0.98–0.99.

## multigrain/
`multigrain_score_5domains.csv`: the pre-registered frozen multi-grain probe, scored for five hierarchies:
speech (sex → speaker), machine (type → unit), bird (species → source recording), city (class → source), music (genre → artist).
Fine classes need at least 10 clips. The pre-registered criterion is not met in any domain: no Kruskal–Wallis test of
organization by training paradigm reaches p < 0.05. In bird, the source-recording level is close to near-duplicate matching,
because clips cut from one recording are near-duplicates.

## self_rsa_model_level/
Model-level (seeds averaged) self-RSA links.

- `first_link_model_level.csv`: uniformity → self-RSA. For the top-K cells (n = 4 models) `p_model` is the
  **exact** two-sided permutation p over all 24 orderings; the smallest attainable value is 0.083. The scipy asymptotic p is
  not valid at n = 4 (for example, bird top-K: asymptotic 0.0, exact 0.083).
  It includes the vessel cells (aggregate correlations only).
- The second link (partial self-RSA → gain, controlling for uniformity, with gain from the converged probe) for the
  public domains is computed by `scripts/paper/finding3_reorganization.py` and reported in `APPENDIX.md` §F.
- After Benjamini–Hochberg correction, the first link survives in none of the 18 cells and the second link in none of the public-domain cells.

## Code
- `scripts/additional_analyses/mls_crossling_rsa.py --language <lang>`: per-language RSA matrix and summary.
- `scripts/additional_analyses/multigrain_ext_probe.py` and `multigrain_ext_score.py`: multi-grain probe and scoring.
- Paths come from `AUDIO_COMP_ROOT` (repo root, default `~/audio_comp`) and `AUDIO_COMP_SCRATCH` (embeddings, default `~/scratch/audio_comp`).
- The BH columns and the exact n = 4 permutation p values are computed from these outputs as described above.
