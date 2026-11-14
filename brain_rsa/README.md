# Brain RSA detour

Side comparison, kept separate from the main Phase 1/Stage pipeline: how well
do our existing foundation-model embeddings predict human auditory-cortex
fMRI responses, using Tuckute, Feather, Boebinger & McDermott (2023), *Many
but not all deep neural network audio models capture brain responses...*,
PLoS Biology (repo: `gretatuckute/auditory_brain_dnn`).

## What's here

- `auditory_brain_dnn/` — cloned upstream repo (unmodified). Its `data/`
  subfolder (downloaded via its own `setup_utils/download_files.py`,
  `get_data=True`) has the 165-sound stimulus set
  (`data/stimuli/165_natural_sounds_16kHz/`) and the NH2015 (Norman-Haignere
  et al. 2015, 7,694 voxels/8 participants), B2021, and NH2015comp neural
  datasets under `data/neural/`.
- `extract_activations.py` / `extract_activations.sbatch` — extracts each
  model's pooled final-layer embedding (same `encoder.embed()` call the main
  pipeline's `extract_embeddings.py` uses) on the 165 stimuli, one model per
  Slurm job, output to `activations/<model>.npz`.
- `run_rsa.py` — computes RSA between each model's embedding correlation
  matrix and the NH2015 brain correlation matrix, per participant, reusing
  the paper's own correlation-matrix / RSA math imported directly from
  `auditory_brain_dnn/aud_dnn/analyze/rsa_matrix_calculation_all_models.py`
  (`run_correlation_on_feature_matrix`, `correlate_two_matrices_rsa`) rather
  than reimplementing it, to avoid a methodology mismatch. Also reports the
  leave-one-out participant noise ceiling and each model's noise-corrected
  RSA score. Writes `results_final_layer_nh2015.csv`.

## v1 scope (this pass)

- **Final layer only** — one pooled embedding per model, not per-layer.
  The paper's actual finding is about *layer-wise* correspondence (middle
  layers -> primary auditory cortex, deep layers -> non-primary cortex), so
  this v1 can at best show whether a model's final representation tracks
  the brain overall, not reproduce the paper's core layer-stage result.
  Per-layer extraction (via `output_hidden_states=True`) is the natural
  next step once this quick pass gives a sane, sanity-checkable number.
- **NH2015 only** (not B2021), **whole-brain** (no ROI split), **no
  train/test cross-validation** — appropriate simplifications when there's
  only one embedding per model to evaluate (no "best layer" to choose via
  held-out data).
- Models: a handful of our already-registered speech-domain adapters
  (`wav2vec2`, `hubert`, `wavlm`, `whisper`, `ast`) — `wav2vec` and `AST` are
  both in the original paper's own external-model list, useful as a rough
  sanity check even though our checkpoints/adapters aren't identical to
  theirs.

## Running it

```bash
# one Slurm job per model
for m in wav2vec2 hubert wavlm whisper ast; do
  sbatch --export=MODEL=$m --job-name=brain_rsa_$m brain_rsa/extract_activations.sbatch
done

# once all activations/*.npz exist:
python brain_rsa/run_rsa.py
```

## Relationship to the main project

This is an exploratory side comparison against an external, independently
collected ground truth (human fMRI), not part of the Phase 1/Stage roadmap
in `CLAUDE.md`. Kept in its own top-level directory so it doesn't get
conflated with the RSA/CKA cross-model geometry work, which compares models
against *each other*, not against brain data.
