"""Frozen multi-grain probe for bird / city / music (criterion specified before any run). Same protocol as the speech
(sex -> speaker) and machine (type -> unit) hierarchies; label hierarchies:
  bird : coarse = species; fine = species/source recording (clip id .../birdclef_clips/<species>/<rec>_<k>.wav)
  city : coarse = classID; fine = classID/fsID (urbansound8k_manifest.csv joined on row_index = clip-id prefix)
  music: coarse = genre (fma_genre_manifest); fine = genre/artist_id (FMA tracks.csv, sha256 f73260fd...)
Fine labels with < 10 clips (among clips present in the embeddings) are dropped, with their clips, before splitting.
Output: results/additional/multigrain_parts/{domain}__{model}.csv"""
import argparse, os, re, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
warnings.filterwarnings("ignore")
EMBS = [os.environ.get("AUDIO_COMP_SCRATCH", os.path.expanduser("~/scratch/audio_comp")) + "/finetune_data_geometry_emb", os.environ.get("AUDIO_COMP_ROOT", os.path.expanduser("~/audio_comp")) + "/results/finetune_data_geometry_emb"]
OUT = os.environ.get("AUDIO_COMP_ROOT", os.path.expanduser("~/audio_comp")) + "/results/additional/multigrain_parts"
D = os.environ.get("AUDIO_COMP_ROOT", os.path.expanduser("~/audio_comp")) + "/scripts/echo_b7_multigrain_ext"
DS = {"bird": "birdclef", "city": "urbansound8k", "music": "fma_genre"}
CS = [0.01, 0.1, 1.0, 10.0]; MIN_FINE = 10

def labels(domain, ids):
    if domain == "bird":
        sp = [p.split("birdclef_clips/")[1].split("/")[0] for p in ids]
        rec = [os.path.basename(p).split("_")[0] for p in ids]
        return np.array(sp), np.array([f"{s}/{r}" for s, r in zip(sp, rec)])
    if domain == "city":
        man = pd.read_csv(f"{D}/urbansound8k_manifest.csv").set_index("row_index")
        ri = [int(os.path.basename(p).split("_")[0]) for p in ids]
        cls = man.loc[ri, "classID"].astype(str).values; fs = man.loc[ri, "fsID"].astype(str).values
        return cls, np.array([f"{c}/{s}" for c, s in zip(cls, fs)])
    mp = pd.read_csv(f"{D}/fma_small_artist_map.csv").set_index("track_id")
    tid = [int(re.search(r"(\d+)\.\w+$", p).group(1)) for p in ids]
    g = mp.loc[tid, "genre"].astype(str).values; ar = mp.loc[tid, "artist_id"].astype(int).astype(str).values
    return g, np.array([f"{x}/{y}" for x, y in zip(g, ar)])

def norm_acc(acc, y_te):
    ch = pd.Series(y_te).value_counts(normalize=True).iloc[0]; return (acc - ch) / (1 - ch)

ap = argparse.ArgumentParser(); ap.add_argument("--domain", required=True, choices=list(DS)); ap.add_argument("--model", required=True); a = ap.parse_args()
f = next(f"{d}/{DS[a.domain]}__{a.model}.npz" for d in EMBS if os.path.exists(f"{d}/{DS[a.domain]}__{a.model}.npz"))
z = np.load(f, allow_pickle=True); X = z["embeddings"].astype(np.float64); ids = [str(c) for c in z["clip_ids"]]
coarse, fine = labels(a.domain, ids)
vc = pd.Series(fine).value_counts(); keep = np.isin(fine, vc[vc >= MIN_FINE].index)
X, coarse, fine = X[keep], coarse[keep], fine[keep]
print(f"{a.domain}/{a.model}: {keep.sum()} clips kept of {len(keep)}; coarse {len(np.unique(coarse))}, fine {len(np.unique(fine))}", flush=True)
os.makedirs(OUT, exist_ok=True); rows = []; bestC = {}
for rs in range(5):
    idx = np.arange(len(X))
    tr, te = train_test_split(idx, test_size=0.2, stratify=fine, random_state=rs)
    tr, va = train_test_split(tr, test_size=0.25, stratify=fine[tr], random_state=rs)
    sc = StandardScaler().fit(X[tr]); Xtr, Xva, Xte = sc.transform(X[tr]), sc.transform(X[va]), sc.transform(X[te])
    for level, y in [("coarse", coarse), ("fine", fine)]:
        if rs == 0:
            bestC[level] = max(CS, key=lambda C: LogisticRegression(C=C, max_iter=5000).fit(Xtr, y[tr]).score(Xva, y[va]))
        lr = LogisticRegression(C=bestC[level], max_iter=5000).fit(Xtr, y[tr])
        lin = lr.score(Xte, y[te]); knn = KNeighborsClassifier(n_neighbors=10).fit(Xtr, y[tr]).score(Xte, y[te])
        rows.append(dict(domain=a.domain, model=a.model, resplit=rs, level=level, n_classes=len(np.unique(y)), n_clips=len(X), C=bestC[level],
                         n_iter=int(np.max(lr.n_iter_)), lin=lin, knn=knn, lin_norm=norm_acc(lin, y[te]), knn_norm=norm_acc(knn, y[te])))
        print(rows[-1], flush=True)
        pd.DataFrame(rows).to_csv(f"{OUT}/{a.domain}__{a.model}.csv", index=False)
