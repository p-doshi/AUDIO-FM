"""Step 5 - model response profile clustering at the clip-pair level.

For each clip pair (i,j) the 19-dim vector of model dissimilarity scores is
its "model response profile". k-means with k in {5,10,15}; silhouette score
(on a 20k-pair subsample for tractability) reported for each.

For each cluster of the k=10 solution:
  - dominant domain-label pairs (category_i x category_j)
  - physics feature groups with the strongest between-cluster separation:
    one-way ANOVA of each group's pair-distance across clusters,
    Benjamini-Hochberg corrected.

Saves results/physics_pair_clusters.csv (per-pair cluster assignment for
k=10, with the two category labels and the 8 physics group distances) and
prints the cluster summaries.
"""
import numpy as np, pandas as pd
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics import silhouette_score
from scipy.stats import f_oneway
import common as C

RNG = 42


def main():
    tensor = np.load(f"{C.OUT}/rdm_tensor.npy")
    meta = np.load(f"{C.OUT}/rdm_tensor_meta.npz", allow_pickle=True)
    clip_order = np.load(f"{C.OUT}/physics_rdms/_clip_order.npy", allow_pickle=True)
    tpos = {c: i for i, c in enumerate(meta["clip_ids"])}
    keep = np.array([tpos[c] for c in clip_order])
    cats = meta["categories"][keep]
    n = len(keep)
    iu = np.triu_indices(n, 1)
    npairs = len(iu[0])

    # (npairs, 19) model response profiles
    prof = np.zeros((npairs, len(C.MODELS)), dtype=np.float32)
    for k in range(len(C.MODELS)):
        sub = tensor[np.ix_(keep, keep, [k])][:, :, 0]
        prof[:, k] = sub[iu]
    Z = (prof - prof.mean(0)) / (prof.std(0) + 1e-12)
    print(f"[step5] pair profiles {Z.shape}")

    # pair-level physics group distances
    phys = {}
    for g in C.FEATURE_GROUPS:
        phys[g] = np.load(f"{C.OUT}/physics_rdms/{g}.npy")[iu]

    cat_i = cats[iu[0]]; cat_j = cats[iu[1]]
    pair_lab = np.array(["|".join(sorted([a, b])) for a, b in zip(cat_i, cat_j)])

    rng = np.random.RandomState(RNG)
    sil_idx = rng.choice(npairs, size=min(20000, npairs), replace=False)

    results = {}
    for k in (5, 10, 15):
        km = MiniBatchKMeans(n_clusters=k, random_state=RNG, n_init=10,
                             batch_size=4096, max_iter=300)
        lab = km.fit_predict(Z)
        sil = silhouette_score(Z[sil_idx], lab[sil_idx])
        results[k] = (lab, sil)
        print(f"  k={k:2d}  silhouette={sil:.4f}  "
              f"cluster sizes={np.bincount(lab).tolist()}")

    lab10 = results[10][0]
    out = pd.DataFrame({"cluster": lab10, "cat_i": cat_i, "cat_j": cat_j,
                        "pair_label": pair_lab})
    for g in C.FEATURE_GROUPS:
        out[f"dist_{g}"] = phys[g]
    out.to_csv(f"{C.OUT}/physics_pair_clusters.csv", index=False)

    print("\n[step5] k=10 cluster summaries")
    for c in range(10):
        m = lab10 == c
        vc = pd.Series(pair_lab[m]).value_counts(normalize=True).head(4)
        print(f"\n cluster {c}  (n={m.sum()}, {100*m.mean():.1f}% of pairs)")
        for lbl, frac in vc.items():
            print(f"    {lbl:30s} {100*frac:5.1f}%")

    print("\n[step5] between-cluster separation per physics group "
          "(one-way ANOVA across the 10 clusters, BH-corrected)")
    fstats, pvals = [], []
    for g in C.FEATURE_GROUPS:
        groups = [phys[g][lab10 == c] for c in range(10)]
        F, p = f_oneway(*groups)
        fstats.append(F); pvals.append(p)
    rej, q = C.bh(pvals)
    tab = pd.DataFrame({"feature_group": C.FEATURE_GROUPS, "F": fstats,
                        "p": pvals, "q_bh": q, "sig": rej}
                       ).sort_values("F", ascending=False)
    print(tab.to_string(index=False))
    print(f"\n[step5] saved {C.OUT}/physics_pair_clusters.csv")


if __name__ == "__main__":
    main()
