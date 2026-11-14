"""Step 6 - per-model residual analysis.

For each of the 19 models: rank all clip pairs by |z(model RDM) - z(combined
physics RDM)| on the upper triangle, and print the 20 pairs where the model
disagrees most with wave-propagation physics (physics says similar / model
says dissimilar, or vice versa). Prints domain labels + clip IDs + signed
residual. Saves results/physics_residuals_per_model.csv.
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata
import common as C

TOPK = 20


def zr(x):
    r = rankdata(x)
    return (r - r.mean()) / r.std()


def main():
    tensor = np.load(f"{C.OUT}/rdm_tensor.npy")
    meta = np.load(f"{C.OUT}/rdm_tensor_meta.npz", allow_pickle=True)
    clip_order = np.load(f"{C.OUT}/physics_rdms/_clip_order.npy", allow_pickle=True)
    tpos = {c: i for i, c in enumerate(meta["clip_ids"])}
    keep = np.array([tpos[c] for c in clip_order])
    ids = meta["clip_ids"][keep]
    cats = meta["categories"][keep]
    n = len(keep)
    iu = np.triu_indices(n, 1)

    phys_all = np.load(f"{C.OUT}/physics_rdms/_ALL.npy")[iu]
    zphys = zr(phys_all)

    rows = []
    for k, mdl in enumerate(C.MODELS):
        sub = tensor[np.ix_(keep, keep, [k])][:, :, 0][iu]
        zmod = zr(sub)
        resid = zmod - zphys                         # + : model >> physics
        top = np.argsort(-np.abs(resid))[:TOPK]
        print(f"\n=== {mdl}  (top {TOPK} model-vs-physics disagreements) ===")
        print(f"{'clip_i':<26}{'clip_j':<26}{'cat_i':<14}{'cat_j':<14}"
              f"{'resid':>8}{'m_z':>7}{'p_z':>7}")
        for t in top:
            i, j = iu[0][t], iu[1][t]
            print(f"{ids[i]:<26}{ids[j]:<26}{cats[i]:<14}{cats[j]:<14}"
                  f"{resid[t]:>8.2f}{zmod[t]:>7.2f}{zphys[t]:>7.2f}")
            rows.append(dict(model=mdl, clip_i=ids[i], clip_j=ids[j],
                             cat_i=cats[i], cat_j=cats[j],
                             residual=resid[t], model_z=zmod[t],
                             physics_z=zphys[t],
                             direction="model_more_dissimilar" if resid[t] > 0
                             else "physics_more_dissimilar"))
    pd.DataFrame(rows).to_csv(f"{C.OUT}/physics_residuals_per_model.csv", index=False)
    print(f"\n[step6] saved {C.OUT}/physics_residuals_per_model.csv")


if __name__ == "__main__":
    main()
