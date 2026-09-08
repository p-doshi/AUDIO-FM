"""Step 17 - what explains the ~63% of cross-model geometry that the 8
physics feature groups do NOT?

For each of the 19 models on the Step-1 1000-clip set:
   raw RDM (rank)  =  physics-fitted part  +  residual
where physics-fitted = OLS of rank(raw RDM) on the 8 rank physics-group RDMs
(the Step-7 decomposition).

Then attribute the residual (and the raw geometry) to a hierarchy of
explanatory RDMs, each rank-transformed, via incremental R^2:

  1. physics        - the 8 groups (baseline, ~Step 4/7)
  2. coarse category - same/different among the 8 probe domains
  3. fine subclass   - same/different speaker | genre | species | ... where derivable
  4. extended timbre - MFCC + spectral contrast + chroma + centroid/rolloff/
                       flatness/bandwidth/zcr/rms  (standard descriptors NOT in
                       the physics set), z-scored, Euclidean RDM
  5. duration        - clip-length RDM (nuisance / confound check)
  -> unexplained     - 1 - R^2(all of the above)

Also: PCA of the 19 residual RDMs -> is there a *second* shared axis, and what
does it correlate with?

Output: results/physics_residual_breakdown.csv , results/physics_residual_pca.csv
"""
import os, re, numpy as np, pandas as pd, librosa
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import squareform, pdist
import common as C
from step15_subclass_probe import subclass_labels

REPO = C.REPO
OUT = C.OUT


def z(a): return (a - a.mean(0)) / (a.std(0) + 1e-9)
def zr(v): return z(rankdata(v))


def r2_of(y, cols):
    """R^2 of y regressed on the given predictor columns (each 1-D, upper-tri)."""
    if not cols:
        return 0.0
    X = np.column_stack([z(c) for c in cols])
    b = np.linalg.pinv(X.T @ X) @ (X.T @ y)
    return float(1 - np.var(y - X @ b) / np.var(y))


def main():
    tensor = np.load(f"{OUT}/rdm_tensor.npy")
    meta = np.load(f"{OUT}/rdm_tensor_meta.npz", allow_pickle=True)
    ids = np.array([str(x) for x in meta["clip_ids"]])
    cats = np.array([str(x) for x in meta["categories"]])
    n = len(ids); iu = np.triu_indices(n, 1)
    co = list(np.load(f"{OUT}/physics_rdms/_clip_order.npy", allow_pickle=True))
    assert co == list(ids), "physics_rdms clip order != rdm_tensor order"

    # ---- physics group predictors (8) ----
    phys_cols = [rankdata(np.load(f"{OUT}/physics_rdms/{g}.npy")[iu]) for g in C.FEATURE_GROUPS]

    # ---- coarse category predictor ----
    cat_i, cat_j = cats[iu[0]], cats[iu[1]]
    same_cat = (cat_i == cat_j).astype(float)
    # also a full category-pair design (one column per unordered pair) for a
    # richer "semantic domain" predictor
    pairkey = np.array(["|".join(sorted([a, b])) for a, b in zip(cat_i, cat_j)])
    catpair_cols = [(pairkey == k).astype(float) for k in np.unique(pairkey)]

    # ---- fine subclass predictors ----
    tasks = subclass_labels(ids, cats)
    fine_cols = []
    fine_report = {}
    lab_full = {}
    for task, (sel, y) in tasks.items():
        lab = np.array(["__NA__"] * n, dtype=object)
        lab[sel] = y
        li, lj = lab[iu[0]], lab[iu[1]]
        valid = (li != "__NA__") & (lj != "__NA__")
        same = np.where(valid, (li == lj).astype(float), 0.0)
        # centre within the valid mask so "NA" pairs contribute nothing
        if valid.sum() > 100 and same[valid].std() > 1e-6:
            col = np.zeros(len(iu[0]))
            col[valid] = same[valid] - same[valid].mean()
            fine_cols.append(col)
            fine_report[task] = int(valid.sum())
    lab_full = None

    # ---- extended timbre RDM (standard descriptors NOT in the physics set) ----
    m = pd.read_csv(C.MANIFEST).set_index("clip_id")
    feats = []
    print(f"[step17] extended-timbre features for {n} clips ...")
    for k, cid in enumerate(ids):
        y, sr = librosa.load(os.path.join(C.DATA_ROOT, m.loc[cid, "path"]),
                             sr=22050, mono=True, duration=10.0)
        if not np.any(y):
            feats.append(np.zeros(49)); continue
        y = librosa.util.normalize(y)
        S = np.abs(librosa.stft(y, n_fft=1024, hop_length=512)) + 1e-9
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(1)
        contr = librosa.feature.spectral_contrast(S=S, sr=sr).mean(1)      # 7
        chroma = librosa.feature.chroma_stft(S=S**2, sr=sr).mean(1)        # 12
        cent = librosa.feature.spectral_centroid(S=S, sr=sr).mean()
        bw = librosa.feature.spectral_bandwidth(S=S, sr=sr).mean()
        roll = librosa.feature.spectral_rolloff(S=S, sr=sr).mean()
        flat = librosa.feature.spectral_flatness(S=S).mean()
        zcr = librosa.feature.zero_crossing_rate(y).mean()
        rms = librosa.feature.rms(y=y)
        try:
            tempo = float(librosa.feature.rhythm.tempo(y=y, sr=sr)[0])
        except Exception:
            tempo = 0.0
        feats.append(np.concatenate([mfcc, contr, chroma,
                                     [cent, bw, roll, flat, zcr, rms.mean(), rms.std(), tempo]]))
        if (k + 1) % 250 == 0:
            print(f"   {k+1}/{n}")
    F = np.nan_to_num(np.vstack(feats))
    timbre_rdm = squareform(pdist(z(F), metric="euclidean"))[iu]
    timbre_col = rankdata(timbre_rdm)

    # ---- duration ----
    dur = m.loc[ids, "duration_sec"].values.astype(float)
    dur_col = np.abs(dur[iu[0]] - dur[iu[1]])

    # ---- per-model decomposition ----
    blocks = {
        "physics": phys_cols,
        "category": [same_cat] + catpair_cols,
        "fine_subclass": fine_cols,
        "ext_timbre": [timbre_col],
        "duration": [dur_col],
    }
    order = ["physics", "category", "fine_subclass", "ext_timbre", "duration"]
    rows = []
    resid_stack = []
    for q, mdl in enumerate(C.MODELS):
        y = zr(tensor[:, :, q][iu])
        # physics-fitted + residual (Step-7 style, physics only)
        Xp = np.column_stack([z(c) for c in phys_cols])
        bp = np.linalg.pinv(Xp.T @ Xp) @ (Xp.T @ y)
        resid = y - Xp @ bp
        resid_stack.append(z(resid))

        rec = {"model": mdl}
        cum, prev = [], 0.0
        for blk in order:
            cum += blocks[blk]
            r2 = r2_of(y, cum)
            rec[f"cumR2_{blk}"] = round(r2, 3)
            rec[f"delta_{blk}"] = round(r2 - prev, 3)
            prev = r2
        rec["unexplained"] = round(1 - prev, 3)
        # residual attribution: how much of the (physics-removed) residual do
        # category / fine / timbre explain
        rec["resid_R2_category"] = round(r2_of(resid, blocks["category"]), 3)
        rec["resid_R2_fine"] = round(r2_of(resid, blocks["fine_subclass"]), 3)
        rec["resid_R2_timbre"] = round(r2_of(resid, blocks["ext_timbre"]), 3)
        rec["resid_R2_cat+fine+timbre"] = round(
            r2_of(resid, blocks["category"] + blocks["fine_subclass"] + blocks["ext_timbre"]), 3)
        rows.append(rec)

    df = pd.DataFrame(rows)
    df.to_csv(f"{OUT}/physics_residual_breakdown.csv", index=False)

    # ---- PCA of the 19 residual RDMs ----
    Rm = np.column_stack(resid_stack)
    U, S, Vt = np.linalg.svd(Rm - Rm.mean(0), full_matrices=False)
    ev = S**2 / (S**2).sum()
    pc = (Rm - Rm.mean(0)) @ Vt.T
    prows = []
    for p in range(4):
        prows.append(dict(
            component=f"PC{p+1}", var_explained=round(float(ev[p]), 3),
            corr_same_category=round(spearmanr(pc[:, p], same_cat).statistic, 3),
            corr_fine_subclass=round(spearmanr(pc[:, p],
                np.sum(blocks["fine_subclass"], axis=0) if blocks["fine_subclass"] else np.zeros(len(iu[0]))).statistic, 3),
            corr_ext_timbre=round(spearmanr(pc[:, p], timbre_col).statistic, 3),
            corr_duration=round(spearmanr(pc[:, p], dur_col).statistic, 3),
        ))
    pd.DataFrame(prows).to_csv(f"{OUT}/physics_residual_pca.csv", index=False)

    # ---- console summary ----
    pd.set_option("display.width", 240, "display.max_columns", 30)
    print("\n=== incremental R^2 on the RAW cross-model geometry (per model) ===")
    print(df[["model", "delta_physics", "delta_category", "delta_fine_subclass",
              "delta_ext_timbre", "delta_duration", "unexplained"]].to_string(index=False))
    print("\n=== mean over 19 models ===")
    for c in ["delta_physics", "delta_category", "delta_fine_subclass", "delta_ext_timbre",
              "delta_duration", "unexplained"]:
        print(f"   {c:22s} {df[c].mean():.3f}")
    print("\n=== residual (physics removed) attribution, mean over models ===")
    for c in ["resid_R2_category", "resid_R2_fine", "resid_R2_timbre", "resid_R2_cat+fine+timbre"]:
        print(f"   {c:26s} {df[c].mean():.3f}")
    print(f"\n   fine-subclass tasks used: {fine_report}")
    print("\n=== PCA of the 19 residual RDMs ===")
    print(pd.DataFrame(prows).to_string(index=False))
    print(f"\n[step17] saved {OUT}/physics_residual_breakdown.csv, {OUT}/physics_residual_pca.csv")


if __name__ == "__main__":
    main()
