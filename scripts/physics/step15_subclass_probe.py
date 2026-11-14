"""Step 15 - fine-grained WITHIN-domain subclass probing.

For each probe-set domain, derive a subclass label for the exact clips in the
Step-8 cache and run a 5-fold linear probe (logistic regression) on:
  distilled_phys      - Step-9 RDM-distilled model embedding (128-d)
  distilled_phys_mel  - Step-9 phys+mel distilled embedding (128-d)
  phys_frontend       - fixed 65-d PC1 frontend, mean-pooled
  logmel              - 48-d log-mel, mean-pooled
  <each of 19 foundation models> - frozen embeddings (native dim)

This is the direct test of whether the geometry-distilled model retains any
fine-grained information, and how far it is from the teachers on the tasks
the coarse consensus geometry does NOT contain.

Output: results/physics_subclass_probe.csv
"""
import os, re, numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
import common as C
from step8_phys_arch import CACHE

CKPT = C.OUT
FMA_TRACKS = "/scratch/user/audio_comp/home_migrated/audio_comp_data/raw/fma_metadata/tracks.csv"
US8K_MANIFEST = f"{C.REPO}/data/urbansound8k_manifest.csv"
MIN_PER_CLASS = 8


class Embed(nn.Module):
    def __init__(self, in_dim, emb=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, emb, 5, stride=2, padding=2), nn.ReLU())

    def forward(self, x):
        return self.net(x.transpose(1, 2)).mean(-1)


def distilled_embed(ckpt_name, phys, mel):
    ck = torch.load(f"{CKPT}/physics_distill_ckpt_{ckpt_name}.pt", map_location="cpu")
    m = Embed(ck["in_dim"], ck["emb"]); m.load_state_dict(ck["state_dict"]); m.eval()
    zp = (np.array(ck["znorm"]["phys"][0], np.float32), np.array(ck["znorm"]["phys"][1], np.float32))
    zm = (np.array(ck["znorm"]["mel"][0], np.float32), np.array(ck["znorm"]["mel"][1], np.float32))
    P = (phys - zp[0]) / zp[1]
    if ck["in_dim"] > P.shape[2]:
        X = np.concatenate([P, (mel - zm[0]) / zm[1]], axis=2).astype(np.float32)
    else:
        X = P.astype(np.float32)
    with torch.no_grad():
        return np.vstack([m(torch.tensor(X[i:i+256])).numpy() for i in range(0, len(X), 256)])


def subclass_labels(ids, cats):
    """dict domain -> (mask_into_ids, label_array, task_name)."""
    out = {}
    idx_by_cat = {c: np.where(cats == c)[0] for c in set(cats)}

    # music -> FMA genre_top
    tr = pd.read_csv(FMA_TRACKS, index_col=0, header=[0, 1], low_memory=False)
    genre = tr[("track", "genre_top")].to_dict()
    mi = idx_by_cat["music"]
    lab = np.array([genre.get(int(ids[i].split("/")[1]), None) for i in mi], dtype=object)
    keep = lab != None
    out["music/genre"] = (mi[keep], lab[keep].astype(str))

    # speech -> LibriSpeech speaker
    mi = idx_by_cat["speech"]
    out["speech/speaker"] = (mi, np.array([ids[i].split("/")[1].split("-")[0] for i in mi]))

    # speech_noisy -> AMI speaker token (AMI_<mtg>_sdm_<SPK>_<..>)
    mi = idx_by_cat["speech_noisy"]
    spk = [re.search(r"sdm_([A-Z]+\d+)_", ids[i]) for i in mi]
    lab = np.array([s.group(1) if s else "NA" for s in spk])
    keep = lab != "NA"
    out["speech_noisy/speaker"] = (mi[keep], lab[keep])

    # music_noisy -> SingVERSE artist (first "_"-token)
    mi = idx_by_cat["music_noisy"]
    out["music_noisy/singer"] = (mi, np.array([ids[i].split("/")[1].split("_")[0] for i in mi]))

    # machine_sounds -> MIMII machine type  and type+id
    mi = idx_by_cat["machine_sounds"]
    out["machine/type"] = (mi, np.array([ids[i].split("/")[1].split("_")[0] for i in mi]))
    out["machine/type_condition"] = (mi, np.array([
        "_".join([ids[i].split("/")[1].split("_")[0],
                  "abn" if "abnormal" in ids[i] else "nrm"]) for i in mi]))

    # city_noise -> UrbanSound8K soundevent via row_index join
    try:
        u = pd.read_csv(US8K_MANIFEST).set_index("row_index")["soundevent"].to_dict()
        mi = idx_by_cat["city_noise"]
        lab = np.array([u.get(int(ids[i].split("/")[1]), "NA") for i in mi])
        keep = lab != "NA"
        if keep.sum() > 50:
            out["city_noise/event"] = (mi[keep], lab[keep])
    except Exception as e:
        print(f"  city_noise join skipped: {e}")

    return out


def foundation_embeddings(want_ids):
    out = {}
    for mdl in C.MODELS:
        d = np.load(f"{C.EMB_DIR}/{mdl}.npz", allow_pickle=True)
        pos = {c: i for i, c in enumerate(d["clip_ids"].tolist())}
        out[mdl] = d["embeddings"][[pos[c] for c in want_ids]].astype(np.float64)
    return out


def probe(X, y):
    y = np.asarray(y)
    vc = pd.Series(y).value_counts()
    ok = vc[vc >= MIN_PER_CLASS].index
    m = np.isin(y, ok)
    if m.sum() < 30 or len(ok) < 2:
        return np.nan, len(ok), int(m.sum())
    Xs, ys = X[m], y[m]
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=0)
    accs = []
    for tr, te in skf.split(Xs, ys):
        sc = StandardScaler().fit(Xs[tr])
        clf = LogisticRegression(max_iter=3000, C=1.0, class_weight="balanced")
        clf.fit(sc.transform(Xs[tr]), ys[tr])
        accs.append(clf.score(sc.transform(Xs[te]), ys[te]))
    return float(np.mean(accs)), len(ok), int(m.sum())


def main():
    d = np.load(CACHE, allow_pickle=True)
    ids = np.array([str(x) for x in d["clip_ids"]]); cats = np.array([str(x) for x in d["categories"]])
    phys, mel = np.nan_to_num(d["phys"]), np.nan_to_num(d["mel"])
    print(f"[step15] {len(ids)} clips")

    feats = {
        "distilled_phys": distilled_embed("phys", phys, mel),
        "distilled_phys_mel": distilled_embed("phys_mel", phys, mel),
        "phys_frontend": np.nan_to_num(phys).mean(1),
        "logmel": mel.mean(1),
    }
    fnd = foundation_embeddings(list(ids))

    tasks = subclass_labels(ids, cats)
    rows = []
    for task, (sel, y) in tasks.items():
        nchance = 1.0 / len(set(y))
        rec = {"task": task, "n_clips": len(sel), "n_classes_raw": len(set(y))}
        for fname, F in feats.items():
            acc, ncls, nused = probe(F[sel], y)
            rec[fname] = round(acc, 3) if acc == acc else np.nan
        fnd_acc = {}
        for mdl, E in fnd.items():
            a, ncls, nused = probe(E[sel], y)
            fnd_acc[mdl] = a
        valid = {k: v for k, v in fnd_acc.items() if v == v}
        rec["n_classes_probed"] = ncls
        rec["chance"] = round(1.0 / ncls, 3) if ncls else np.nan
        rec["foundation_best"] = round(max(valid.values()), 3) if valid else np.nan
        rec["foundation_best_model"] = max(valid, key=valid.get) if valid else "NA"
        rec["foundation_median"] = round(float(np.median(list(valid.values()))), 3) if valid else np.nan
        rec["_fnd_all"] = {k: round(v, 3) for k, v in fnd_acc.items()}
        rows.append(rec)
        print(f"\n {task}  (n={len(sel)}, {ncls} classes probed, chance={rec['chance']})")
        print(f"   distilled_phys={rec['distilled_phys']}  distilled_phys_mel={rec['distilled_phys_mel']}"
              f"  phys_frontend={rec['phys_frontend']}  logmel={rec['logmel']}")
        print(f"   foundation: best={rec['foundation_best']} ({rec['foundation_best_model']})  "
              f"median={rec['foundation_median']}")

    df = pd.DataFrame(rows)
    df.drop(columns=["_fnd_all"]).to_csv(f"{C.OUT}/physics_subclass_probe.csv", index=False)
    pd.DataFrame([{"task": r["task"], **r["_fnd_all"]} for r in rows]).to_csv(
        f"{C.OUT}/physics_subclass_probe_foundation_full.csv", index=False)
    print("\n=== SUMMARY (5-fold linear-probe accuracy) ===")
    cols = ["task", "n_classes_probed", "chance", "distilled_phys", "distilled_phys_mel",
            "phys_frontend", "logmel", "foundation_median", "foundation_best", "foundation_best_model"]
    print(df[cols].to_string(index=False))
    print(f"\n[step15] saved {C.OUT}/physics_subclass_probe.csv")


if __name__ == "__main__":
    main()
