"""Step 3 - one N x N RDM per physics feature group.

Within each feature group: z-score every column across clips, then pairwise
Euclidean distance. Saves results/physics_rdms/<group>.npy plus a combined
RDM (all groups' z-scored columns concatenated) as _ALL.npy.
"""
import os, numpy as np
from scipy.spatial.distance import squareform, pdist
import common as C


def main():
    os.makedirs(f"{C.OUT}/physics_rdms", exist_ok=True)
    F = np.load(f"{C.OUT}/physics_features.npz", allow_pickle=True)
    meta = np.load(f"{C.OUT}/rdm_tensor_meta.npz", allow_pickle=True)

    # align physics-feature rows to the RDM tensor clip order
    fpos = {c: i for i, c in enumerate(F["clip_ids"])}
    order = [fpos[c] for c in meta["clip_ids"] if c in fpos]
    kept_ids = [c for c in meta["clip_ids"] if c in fpos]
    print(f"[step3] {len(kept_ids)}/{len(meta['clip_ids'])} clips have physics features")
    np.save(f"{C.OUT}/physics_rdms/_clip_order.npy", np.array(kept_ids))

    allcols = []
    for g in C.FEATURE_GROUPS:
        A = F[g][order].astype(np.float64)
        Z = (A - A.mean(0)) / (A.std(0) + 1e-12)
        rdm = squareform(pdist(Z, metric="euclidean"))
        np.save(f"{C.OUT}/physics_rdms/{g}.npy", rdm)
        allcols.append(Z)
        iu = np.triu_indices(rdm.shape[0], 1)
        print(f"  {g:16s} feat_shape={A.shape} rdm={rdm.shape} "
              f"mean={rdm[iu].mean():.3f}")
    Zall = np.hstack(allcols)
    rdm_all = squareform(pdist(Zall, metric="euclidean"))
    np.save(f"{C.OUT}/physics_rdms/_ALL.npy", rdm_all)
    print(f"  {'_ALL':16s} feat_shape={Zall.shape} rdm={rdm_all.shape}")
    print(f"[step3] saved {C.OUT}/physics_rdms/")


if __name__ == "__main__":
    main()
