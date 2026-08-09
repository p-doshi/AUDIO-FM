# audio_comp

Cross-model comparison of audio foundation models: build a representational
dissimilarity matrix (RDM) per model over a fixed probe set, compare RDMs
across models via RSA (primary) and CKA (secondary), and use TwoNN intrinsic
dimension as an independent structural check. See `CLAUDE.md` for the full
project scope, decision rule, and Phase 2 (JEPA relational distillation)
plan gated on Phase 1's results.

This repo is meant to be the reusable framework for that comparison, not a
one-off script collection — adding a new model or a new dataset/category
should not require touching the extraction or comparison pipeline.

## Layout

```
audio_comp/
  models/         model adapters — one file per model, registered by name
  data/           dataset sources (one file per dataset) + probe-set builder
  geometry/       RDM / RSA / CKA / TwoNN intrinsic dimension
  pipelines/      extract_embeddings.py (one model at a time), compare_models.py
configs/
  models.yaml     which registered models are active in the current run
  categories.yaml which dataset source backs each probe-set category, and how many clips
scripts/slurm/    sbatch job + submission script (one Slurm job per model, parallel)
data/
  probe_set_manifest.csv   the ONLY probe-set artifact tracked in git — raw
                           audio lives on $SCRATCH, reproducible from this
                           manifest + build_probe_set.py + the category configs
results/          RSA/CKA matrices + heatmaps — the actual Phase 1 deliverable
tests/
journal.md        running lab notebook (CLAUDE.md-mandated, append-only)
```

## Adding a new model

1. Create `audio_comp/models/your_model.py`, subclass `BaseAudioEncoder`
   (`audio_comp/models/base.py`), set `info = ModelInfo(...)`, implement
   `load()` and `embed_batch()`. Decorate the class with
   `@register_model("your_model")`.
2. Import your module from `audio_comp/models/__init__.py`.
3. Add `your_model` to `configs/models.yaml`'s `active_models` list.

If the checkpoint isn't natively `transformers.from_pretrained`-able (see
`musicfm.py`, `audio_jepa.py`, `beats.py` for real examples of this), keep
`load()`/`embed_batch()` raising `NotImplementedError` with a clear message
until the loader is actually wired up and tested — don't guess at loader
code that hasn't been run.

## Adding a new dataset / category

1. Create `audio_comp/data/sources/your_dataset.py`, subclass
   `BaseDatasetSource` (`audio_comp/data/base.py`), set
   `info = DatasetInfo(...)`, implement `iter_clips()` (deterministic given
   a seed). Decorate with `@register_dataset("your_dataset")`.
2. Import your module from `audio_comp/data/sources/__init__.py`.
3. Add an entry under `categories:` in `configs/categories.yaml` pointing
   `source:` at your dataset name.

## Running the pilot end to end

```bash
# one-time environment setup — see "Environment setup" below
python -m audio_comp.data.build_probe_set          # writes data/probe_set_manifest.csv
bash scripts/slurm/submit_all.sh                    # one Slurm job per active model
# ... wait for all extract_embeddings jobs to finish, then:
python -m audio_comp.pipelines.compare_models \
    --embeddings-dir "$SCRATCH/audio_comp/embeddings"
```

`results/` will contain `rsa_matrix.csv`, `cka_matrix.csv`,
`intrinsic_dimension.csv`, and the corresponding heatmap PNGs.

## Environment setup (fir cluster / Digital Research Alliance of Canada)

```bash
# gcc + arrow must be loaded BEFORE the venv is activated (Compute Canada's
# pyarrow — a `datasets` dependency — is provided by the system module, not pip;
# pip install -e . fails with a clear "load the Arrow module first" error otherwise)
module load python/3.11 cuda/12.6 gcc arrow/25.0.0
python -m venv ~/audio-comp-venv
source ~/audio-comp-venv/bin/activate
pip install --no-index torch torchaudio torchvision   # Compute Canada wheelhouse
pip install -e .                                        # everything else, from pyproject.toml
huggingface-cli login                                   # caches your HF token; never paste it into chat
```

`pyarrow` (a `datasets` dependency) comes from the `arrow` module at import
time too, not just at install time — every shell that runs this code (Slurm
jobs, interactive use, `build_probe_set.py`) needs
`module load python/3.11 cuda/12.6 gcc arrow/25.0.0` loaded before activating
the venv. `scripts/slurm/extract_embeddings.sbatch` already does this.

`$SCRATCH/audio_comp/` (not this git repo) holds raw probe-set audio,
downloaded checkpoints, and extracted embeddings — set `AUDIO_COMP_DATA_ROOT`
and `AUDIO_COMP_EXTERNAL` env vars to override the defaults
(`~/audio_comp_data` and `~/audio_comp_external`).

`musicfm` and `audio_jepa` each need one extra one-time step before they'll load:
```bash
bash scripts/setup_musicfm.sh
bash scripts/setup_audio_jepa.sh
```
and `music` category clips need:
```bash
bash scripts/download_fma_small.sh
```

## Current kickoff model set

| Model | Paradigm | License | Status |
|---|---|---|---|
| CLAP (`laion/larger_clap_general`) | Contrastive (audio-text) | Apache-2.0 | active |
| MERT (`m-a-p/MERT-v1-330M`) | Masked modeling (music) | CC-BY-NC-4.0 | active |
| HuBERT (`facebook/hubert-large-ll60k`) | Masked modeling (speech) | Apache-2.0 | active |
| wav2vec 2.0 (`facebook/wav2vec2-large-lv60`) | Masked modeling (speech) | Apache-2.0 | active |
| music2vec (`m-a-p/music2vec-v1`) | data2vec-family, not JEPA (music) | CC-BY-NC-4.0 | active |
| MusicFM (`minzwon/MusicFM`) | Masked modeling (BEST-RQ, music) | MIT | active, needs `scripts/setup_musicfm.sh` |
| Audio-JEPA (`ltuncay/Audio-JEPA`) | JEPA-family (general; **not** the original A-JEPA — see module docstring) | MIT | active, needs `scripts/setup_audio_jepa.sh` |
| BEATs | Masked modeling (general audio) | unverified for weights | deferred, no native HF path |

The original paper's A-JEPA (Fei, Fan, Huang, arXiv 2311.15830) has no
public checkpoint anywhere — `ltuncay/Audio-JEPA` is used as an
explicitly-labeled substitute for the JEPA-family paradigm slot.

**Correction (2026-08-09):** music2vec was originally labeled JEPA-family
here and in CLAUDE.md; it's actually data2vec-family — no separate
predictor network (the architectural line that actually defines JEPA), just
a student encoder predicting an EMA teacher's representations directly. See
the correction note in `CLAUDE.md` and `audio_comp/models/music2vec.py`'s
docstring for the full detail. Practical effect: there are currently zero
working JEPA-family models in the active comparison — `audio_jepa` will be
the first once it's wired up, not a second point to compare against
music2vec.

## Current probe-set categories (pilot: 20 clips/category)

| Category | Source | License |
|---|---|---|
| Music | FMA-small | CC-BY family |
| Speech | LibriSpeech ASR (English) | CC-BY-4.0 |
| Bird sounds | ESC-50 (chirping_birds/crow classes) | CC-BY-NC-3.0 |
| Ship/vessel | DS3500 (ShipsEar-derived) | CC-BY |
| City/urban noise | UrbanSound8K (HF mirror) | CC-BY-NC-4.0 (verify against original) |
