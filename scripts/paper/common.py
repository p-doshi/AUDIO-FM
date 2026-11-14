"""Shared constants and loaders for the paper's analysis scripts.

All scripts read aggregate inputs from RESULTS (default: <repo>/results) and write their
outputs to OUT (default: <repo>/results/paper_outputs). Override with the environment
variables ECHO_RESULTS and ECHO_OUT.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
RESULTS = Path(os.environ.get("ECHO_RESULTS", REPO / "results"))
OUT = Path(os.environ.get("ECHO_OUT", RESULTS / "paper_outputs"))
OUT.mkdir(parents=True, exist_ok=True)

# Public domains, by dataset key. The vessel domain is analysed with the same code on a
# restricted server; only its aggregate statistics are included in results/.
DOMAINS = ["fma_genre", "urbansound8k", "birdclef", "librispeech", "mimii"]
DOMAIN_NAME = {"fma_genre": "music", "urbansound8k": "city", "birdclef": "bird",
               "librispeech": "speech", "mimii": "machine", "vessel": "vessel"}
CHANCE = {"fma_genre": 1 / 8, "urbansound8k": 1 / 10, "birdclef": 1 / 50, "librispeech": 1 / 251, "mimii": 1 / 2}

TOPK_MODELS = ["audio_jepa", "audiomae", "musicfm", "panns_cnn14"]
LORA_MODELS = ["ast", "bird_mae", "clap", "data2vec_audio", "hubert", "mert", "mms", "music2vec", "sew",
               "unispeech_sat", "wav2vec2", "wav2vec2_conformer", "wavlm", "whisper"]
SPEECH_MODELS = ["hubert", "wav2vec2", "wavlm", "unispeech_sat", "sew", "wav2vec2_conformer", "mms",
                 "data2vec_audio", "whisper"]
HOME_DOMAIN = {**{m: "librispeech" for m in SPEECH_MODELS},
               **{m: "fma_genre" for m in ["mert", "musicfm", "music2vec"]}, "bird_mae": "birdclef"}


def primary_condition(model):
    """Adaptation condition used for gain: LoRA for the 14 LoRA-eligible models, top-K for the other 4."""
    return "topk" if model in TOPK_MODELS else "lora"


def difficulty(mean_frozen_acc, chance):
    """Chance-normalized position of a domain's mean frozen accuracy between chance and 1."""
    a, c = mean_frozen_acc, chance
    return (a - c) / (1 - c) if a >= c else (a - c) / c


def load_converged_probe(allora=False, exclude_domains=()):
    """Per-(dataset, model, condition, seed) converged-probe accuracy (frozen, adapted) and gain.
    Uses the whitened Gram-recovered features ("white" columns). Unadapted checkpoints are removed."""
    pat = "converged_probe_gain_*_allora_seed*.csv" if allora else "converged_probe_gain_*_seed*.csv"
    files = [f for f in glob.glob(str(RESULTS / "probes" / "converged" / pat)) if allora or "_allora_" not in f]
    d = pd.concat([pd.read_csv(f) for f in files])
    d = d[d.model != "model"]
    for c in ["seed", "frozen_white", "adapted_white", "gain_white"]:
        d[c] = pd.to_numeric(d[c])
    d["seed"] = d.seed.astype(int)
    bad = pd.read_csv(RESULTS / "probes" / "unadapted_checkpoints.csv")
    bad = set(zip(bad.dataset, bad.condition, bad.model, bad.seed.astype(int)))
    d = d[[(a, c, m, s) not in bad for a, c, m, s in zip(d.dataset, d.condition, d.model, d.seed)]]
    d = d[~d.dataset.isin(exclude_domains)]
    d = d.drop_duplicates(["dataset", "model", "condition", "seed"], keep="last")
    return d.rename(columns={"frozen_white": "frozen_acc", "adapted_white": "adapted_acc", "gain_white": "gain"})


def frozen_geometry():
    return pd.read_csv(RESULTS / "geometry" / "frozen_geometry.csv")


def gain_rows(exclude_domains=()):
    """Rows for the Finding 3 regression: one per (dataset, model, seed), primary condition only."""
    d = load_converged_probe(exclude_domains=exclude_domains)
    d = d[d.condition == d.model.map(primary_condition)]
    uni = {(r.dataset, r.model): r.uniformity for r in frozen_geometry().itertuples()}
    d = d[[(a, m) in uni for a, m in zip(d.dataset, d.model)]].copy()
    d["uniformity"] = [uni[(a, m)] for a, m in zip(d.dataset, d.model)]
    d["domain_match"] = [int(HOME_DOMAIN.get(m) == a) for a, m in zip(d.dataset, d.model)]
    return add_difficulty(d)


def add_difficulty(d):
    d = d.copy()
    m = d.groupby("dataset").frozen_acc.mean()
    d["difficulty"] = d.dataset.map({k: difficulty(v, CHANCE[k]) for k, v in m.items()})
    return d


def read_square(path):
    """Read a model x model matrix and keep only rows/columns present on both axes."""
    q = pd.read_csv(path, index_col=0)
    q = q.loc[q.index.isin(q.columns), [c for c in q.columns if c in q.index]]
    ok = ~q.isna().all(1)
    return q.loc[ok, ok[ok].index]


def bh(p):
    """Benjamini-Hochberg adjusted q-values (step-up)."""
    p = np.asarray(p, float)
    n = len(p)
    o = np.argsort(p)
    q = np.empty(n)
    run = 1.0
    for i in range(n - 1, -1, -1):
        run = min(run, p[o[i]] * n / (i + 1))
        q[o[i]] = run
    return q


def demean_within(d, cols, by="dataset"):
    d = d.copy()
    for c in cols:
        d[c] = d[c] - d.groupby(by)[c].transform("mean")
    return d
