"""Shared setup for the physics reverse-engineering pipeline (steps 1-6).

Design decisions (documented per project convention):

* The full probe set is 16,000 clips. A full 16000 x 16000 x 19 float32 RDM
  tensor is ~19.4 GB and the per-clip physics extraction (EMD in particular)
  would take many hours over 16k clips. We therefore work on a **stratified
  random subsample**: N_PER_CAT clips per category, seeded, drawn only from
  clips that (a) appear in every one of the 19 model embedding files and
  (b) have an audio file on disk with duration >= MIN_DUR_SEC.
  Bump N_PER_CAT and re-run to scale up.//

* The 19 models are exactly configs/models.yaml `active_models`
  (birdnet is excluded: only 10k clips, different clip order).

* Physics features load audio mono at SR Hz, first MAX_DUR_SEC seconds only,
  to bound compute and put every clip on a common time base.
"""
import os, numpy as np, pandas as pd

REPO = "/home/user/audio_comp"
EMB_DIR = "/scratch/user/audio_comp/embeddings"
DATA_ROOT = "/scratch/user/audio_comp/home_migrated/audio_comp_data"
MANIFEST = f"{REPO}/data/probe_set_manifest.csv"
OUT = f"{REPO}/results"

MODELS = ["clap","mert","hubert","wav2vec2","music2vec","musicfm","audio_jepa",
          "panns_cnn14","bird_mae","ast","audiomae","wavlm","whisper",
          "data2vec_audio","mms","unispeech_sat","sew","wav2vec2_conformer","encodecmae"]

N_PER_CAT = int(os.environ.get("N_PER_CAT", "125"))   # -> N = 1000
SEED = 42
SR = 22050
MAX_DUR_SEC = 10.0
MIN_DUR_SEC = 1.0

FEATURE_GROUPS = ["group_delay", "dispersion_fit", "imf_energy", "fractal",
                  "coherence_decay", "radiation", "harmonic", "helmholtz"]


def select_clips():
    """Return a DataFrame (clip_id, category, path, duration_sec) for the
    stratified subsample, in a fixed deterministic order."""
    m = pd.read_csv(MANIFEST)
    ref_ids = np.load(f"{EMB_DIR}/clap.npz", allow_pickle=True)["clip_ids"]
    ref_set = set(ref_ids.tolist())
    m = m[m.clip_id.isin(ref_set)].copy()
    m = m[m.duration_sec >= MIN_DUR_SEC].copy()
    m["abspath"] = m.path.apply(lambda p: os.path.join(DATA_ROOT, p))
    m = m[m.abspath.apply(os.path.exists)].copy()
    rng = np.random.RandomState(SEED)
    parts = []
    for cat, g in m.groupby("category"):
        g = g.sort_values("clip_id")
        take = min(N_PER_CAT, len(g))
        idx = rng.choice(len(g), size=take, replace=False)
        parts.append(g.iloc[np.sort(idx)])
    out = pd.concat(parts).sort_values(["category", "clip_id"]).reset_index(drop=True)
    return out[["clip_id", "category", "abspath", "duration_sec"]]


def model_embeddings(clip_ids):
    """dict model -> (n, d) mean-pooled embeddings aligned to clip_ids order."""
    want = list(clip_ids)
    out = {}
    for mdl in MODELS:
        d = np.load(f"{EMB_DIR}/{mdl}.npz", allow_pickle=True)
        pos = {c: i for i, c in enumerate(d["clip_ids"].tolist())}
        rows = [pos[c] for c in want]
        out[mdl] = d["embeddings"][rows].astype(np.float64)
    return out


def bh(pvals, alpha=0.05):
    """Benjamini-Hochberg. Returns (reject_bool, qvalues)."""
    p = np.asarray(pvals, float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(m) + 1)
    q = np.minimum.accumulate(q[::-1])[::-1]
    out = np.empty(m); out[order] = np.clip(q, 0, 1)
    rej = np.empty(m, bool); rej[order] = q <= alpha
    return rej, out


def upper_tri(mat):
    iu = np.triu_indices(mat.shape[0], k=1)
    return mat[iu]
