"""Step 1 - build the N x N x 19 correlation-distance RDM tensor from
mean-pooled model embeddings. Saves results/rdm_tensor.npy and a sidecar
results/rdm_tensor_meta.npz (clip_ids, categories, model order).

Built fresh from /scratch/.../embeddings/*.npz - the archived
results/archive/ matrices are NOT used.
"""
import numpy as np
from scipy.spatial.distance import squareform, pdist
import common as C


def main():
    clips = C.select_clips()
    n = len(clips)
    print(f"[step1] N={n} clips  ({clips.category.value_counts().to_dict()})")
    embs = C.model_embeddings(clips.clip_id)
    tensor = np.zeros((n, n, len(C.MODELS)), dtype=np.float32)
    for k, mdl in enumerate(C.MODELS):
        # correlation distance = 1 - Pearson r between clip embedding vectors
        rdm = squareform(pdist(embs[mdl], metric="correlation"))
        tensor[:, :, k] = rdm
        iu = np.triu_indices(n, 1)
        print(f"  {mdl:22s} rdm shape={rdm.shape} "
              f"mean={rdm[iu].mean():.3f} min={rdm[iu].min():.3f} max={rdm[iu].max():.3f}")
    np.save(f"{C.OUT}/rdm_tensor.npy", tensor)
    np.savez(f"{C.OUT}/rdm_tensor_meta.npz",
             clip_ids=clips.clip_id.values, categories=clips.category.values,
             models=np.array(C.MODELS))
    print(f"[step1] saved {C.OUT}/rdm_tensor.npy shape={tensor.shape} "
          f"({tensor.nbytes/1e6:.1f} MB)")


if __name__ == "__main__":
    main()
