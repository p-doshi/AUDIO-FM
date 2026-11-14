"""Frozen linear and kNN probes for Finding 2 (requires the cached frozen embeddings).

Embeddings: one <dataset>__<model>.npz per (dataset, model) with keys embeddings, labels, clip_ids, as
written by audio_comp.pipelines.finetune_data_geometry. Directory: $FINETUNE_GEOM_EMB_DIR.

Test split per domain:
  music   FMA-small native split, held-out 'test'
  speech  per-speaker 70/15/15 split, held-out 'test' (closed-set speaker identification)
  city    UrbanSound8K fold 1
  bird    recording-level fold 0 (recording index % 5 == 0), so no held-out recording contributes training clips
  machine stratified 80/20 split (random_state 0)
Features are standardized with a scaler fit on the training clips.
  kNN     k = 10, Euclidean
  linear  logistic regression, C = 1, L-BFGS: max_iter 500 / tol 1e-3 (column `linear`) and run to
          convergence, max_iter 10,000 / tol 1e-6 (column `lin_conv`, used for the kNN - linear gap)
Outputs: results/probes/frozen_knn_linear.csv, results/probes/frozen_linear_iterations.csv
"""
import os
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from common import DOMAINS, REPO, RESULTS

EMB = Path(os.environ.get("FINETUNE_GEOM_EMB_DIR", REPO / "results" / "finetune_data_geometry_emb"))
MODELS = ["ast", "audio_jepa", "audiomae", "bird_mae", "clap", "data2vec_audio", "encodecmae", "hubert", "mert", "mms",
          "music2vec", "musicfm", "panns_cnn14", "sew", "unispeech_sat", "wav2vec2", "wav2vec2_conformer", "wavlm", "whisper"]
LIBRI_MANIFEST = Path(os.environ.get("LIBRISPEECH_SPEAKER_MANIFEST_CSV", REPO / "data" / "librispeech_speaker_manifest.csv"))


def stem(path):
    return re.sub(r"\.(wav|mp3)$", "", str(path).split("/")[-1])


def test_mask(dom, ids, labels):
    ids = [str(x) for x in ids]
    if dom in ("fma_genre", "librispeech"):
        m = pd.read_csv(REPO / "data" / "fma_genre_manifest.csv" if dom == "fma_genre" else LIBRI_MANIFEST)
        sp = {stem(f): s for f, s in zip(m.file, m.split)}
        return np.array([sp.get(stem(i)) == "test" for i in ids])
    if dom == "urbansound8k":
        m = pd.read_csv(REPO / "data" / "urbansound8k_manifest.csv")
        fo = {stem(f): int(x) for f, x in zip(m.file, m.fold)}
        return np.array([fo.get(stem(i)) == 1 for i in ids])
    if dom == "birdclef":
        return np.array([int(re.search(r"/([a-z0-9]+)/(\d+)_\d+\.wav$", i).group(2)) % 5 == 0 for i in ids])
    _, te = next(StratifiedShuffleSplit(1, test_size=0.2, random_state=0).split(np.zeros(len(labels)), labels))
    tm = np.zeros(len(labels), bool)
    tm[te] = True
    return tm


def job(args):
    dom, model = args
    p = EMB / f"{dom}__{model}.npz"
    if not p.exists():
        return None
    z = np.load(p, allow_pickle=True)
    X, y = z["embeddings"].astype(np.float64), z["labels"].astype(int)
    te = test_mask(dom, z["clip_ids"], y)
    sc = StandardScaler().fit(X[~te])
    Xtr, Xte = sc.transform(X[~te]), sc.transform(X[te])
    lin = LogisticRegression(max_iter=500, tol=1e-3, C=1.0).fit(Xtr, y[~te])
    conv = LogisticRegression(max_iter=10000, tol=1e-6, C=1.0).fit(Xtr, y[~te])
    knn = KNeighborsClassifier(n_neighbors=10).fit(Xtr, y[~te])
    return dict(dataset=dom, model=model, n=len(y), n_classes=len(set(y.tolist())), n_test=int(te.sum()),
                linear=lin.score(Xte, y[te]), knn=knn.score(Xte, y[te]), n_iter_linear=int(lin.n_iter_.max()),
                lin_conv=conv.score(Xte, y[te]), n_iter_conv=int(conv.n_iter_.max()))


if __name__ == "__main__":
    with Pool(min(32, os.cpu_count())) as pool:
        rows = [r for r in pool.map(job, [(d, m) for d in DOMAINS for m in MODELS]) if r]
    df = pd.DataFrame(rows).sort_values(["dataset", "model"])
    df[["dataset", "model", "n", "n_classes", "n_test", "linear", "knn"]].to_csv(RESULTS / "probes" / "frozen_knn_linear.csv", index=False)
    df[["dataset", "model", "n_iter_linear", "linear", "n_iter_conv", "lin_conv"]].to_csv(RESULTS / "probes" / "frozen_linear_iterations.csv", index=False)
    print(df.to_string(index=False))
