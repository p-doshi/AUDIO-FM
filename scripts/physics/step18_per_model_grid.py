"""Step 18 - per-model grid, no averaging.

For the Step-16 feature-distilled trunk (consensus1 AND multi19):
  * RSA of its RDM against EACH of the 19 foundation-model RDMs (3000 clips)
  * its probe accuracy on the coarse 8-way domain task and every fine subclass
    task, side by side with EACH of the 19 models on identical folds
  * the delta trunk-minus-model per task -> where the distilled model wins/loses
    against each specific teacher, and whether geometric closeness (RSA) tracks
    skill closeness.

Outputs:
  results/physics_per_model_grid.csv          - one row per representation
  results/physics_distilled_vs_teacher.csv    - trunk RSA + per-task delta vs each of 19
"""
import numpy as np, pandas as pd, torch
from scipy.stats import spearmanr
from scipy.spatial.distance import squareform, pdist
import common as C
from step8_phys_arch import CACHE
from step15_subclass_probe import subclass_labels, probe, foundation_embeddings
import step16_multiteacher as S16

torch.manual_seed(0); np.random.seed(0)
pd.set_option("display.width", 260, "display.max_columns", 40)


def rdm_ut(E):
    n = E.shape[0]
    return squareform(pdist(E, metric="correlation"))[np.triu_indices(n, 1)]


def coarse_probe(E, cats):
    cl = sorted(set(cats)); ci = {c: i for i, c in enumerate(cl)}
    return probe(E, np.array([ci[c] for c in cats]))[0]


def main():
    d = np.load(CACHE, allow_pickle=True)
    ids = np.array([str(x) for x in d["clip_ids"]]); cats = np.array([str(x) for x in d["categories"]])
    phys, mel = np.nan_to_num(d["phys"]), np.nan_to_num(d["mel"])
    tr = np.arange(len(ids))

    def zn(A):
        f = A[tr].reshape(-1, A.shape[2]); return ((A - f.mean(0)) / (f.std(0) + 1e-6)).astype(np.float32)
    X = np.concatenate([zn(phys), zn(mel)], axis=2)

    fnd = foundation_embeddings(list(ids))
    dims = [fnd[m].shape[1] for m in C.MODELS]
    teach_z = [((fnd[m] - fnd[m].mean(0)) / (fnd[m].std(0) + 1e-8)).astype(np.float32) for m in C.MODELS]

    trunk = {}
    for cond in ("consensus1", "multi19"):
        trunk[cond] = S16.train(cond, X, teach_z, dims)

    tasks = subclass_labels(ids, cats)
    task_names = list(tasks.keys())

    # ---- representations to probe ----
    reps = {
        "trunk_consensus1": trunk["consensus1"],
        "trunk_multi19": trunk["multi19"],
        "phys_frontend": np.nan_to_num(phys).mean(1),
        "logmel": mel.mean(1),
    }
    for m in C.MODELS:
        reps[f"model:{m}"] = fnd[m]

    # ---- per-model grid ----
    rows = []
    trunk_rdm = {c: rdm_ut(trunk[c]) for c in trunk}
    for name, E in reps.items():
        rec = {"representation": name, "dim": E.shape[1]}
        rec["coarse_8way"] = round(coarse_probe(E, cats), 3)
        for t in task_names:
            sel, y = tasks[t]
            a = probe(E[sel], y)[0]
            rec[t] = round(a, 3) if a == a else np.nan
        if name.startswith("model:"):
            er = rdm_ut(E)
            rec["RSA_to_trunk_consensus1"] = round(spearmanr(er, trunk_rdm["consensus1"]).statistic, 3)
            rec["RSA_to_trunk_multi19"] = round(spearmanr(er, trunk_rdm["multi19"]).statistic, 3)
        rows.append(rec)
    grid = pd.DataFrame(rows)
    grid.to_csv(f"{C.OUT}/physics_per_model_grid.csv", index=False)

    # ---- distilled vs each teacher ----
    dv = []
    gi = grid.set_index("representation")
    probe_cols = ["coarse_8way"] + task_names
    for m in C.MODELS:
        r = gi.loc[f"model:{m}"]
        row = {"model": m,
               "RSA_to_trunk_multi19": r["RSA_to_trunk_multi19"],
               "RSA_to_trunk_consensus1": r["RSA_to_trunk_consensus1"],
               "model_mean_subclass": round(float(np.nanmean([r[t] for t in task_names])), 3)}
        for c in probe_cols:
            row[f"delta_{c}"] = round(float(gi.loc["trunk_multi19", c] - r[c]), 3)
        dv.append(row)
    dvdf = pd.DataFrame(dv)
    dvdf["trunk_multi19_mean_subclass"] = round(
        float(np.nanmean([gi.loc["trunk_multi19", t] for t in task_names])), 3)
    dvdf.to_csv(f"{C.OUT}/physics_distilled_vs_teacher.csv", index=False)

    # ---- console ----
    print("\n=== PER-MODEL GRID (5-fold linear-probe accuracy) ===")
    show = ["representation", "dim", "coarse_8way"] + task_names + \
           ["RSA_to_trunk_multi19", "RSA_to_trunk_consensus1"]
    print(grid[show].to_string(index=False))

    print("\n=== DISTILLED (multi19) vs EACH TEACHER ===")
    print("  RSA = geometric similarity to the trunk;  delta_* = trunk_acc - model_acc  (+ = trunk wins)")
    cols = ["model", "RSA_to_trunk_multi19", "model_mean_subclass",
            "delta_coarse_8way"] + [f"delta_{t}" for t in task_names]
    print(dvdf[cols].to_string(index=False))

    r = dvdf.dropna(subset=["RSA_to_trunk_multi19", "model_mean_subclass"])
    rho = spearmanr(r["RSA_to_trunk_multi19"], r["model_mean_subclass"]).statistic
    print(f"\n  Spearman(RSA-to-trunk, model's own mean subclass acc) = {rho:.3f}  (n={len(r)})")
    print(f"  trunk_multi19 mean subclass acc = {dvdf['trunk_multi19_mean_subclass'].iloc[0]}")
    print(f"\n[step18] saved results/physics_per_model_grid.csv, results/physics_distilled_vs_teacher.csv")


if __name__ == "__main__":
    main()
