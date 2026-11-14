"""Geometry of the adapted encoder vs the frozen encoder, from the stored Gram matrices.

For every adapted checkpoint of one condition (primary = LoRA / top-K, or ALLoRA) and the frozen encoder on
the same clips (cached embeddings aligned to loader order):
  cosine Gram     C_ij = G_ij / sqrt(G_ii G_jj)   (exact from the uncentred Gram G = E E^T)
  mean_energy     mean(C) = ||mean of L2-normalized embeddings||^2   (the shared offset)
  uniformity      log mean_{i<j} exp(-2 ||z_i - z_j||^2), with ||z_i - z_j||^2 = 2 - 2 C_ij  (Wang & Isola, t = 2)
  pc1_share       top eigenvalue / trace of the double-centred cosine Gram
  part_ratio      participation ratio tr(K)^2 / ||K||_F^2
  hub_cos         skewness of 10-nearest-neighbour occurrence counts under cosine similarity
Adapted checkpoints that are numerically identical to the frozen encoder are skipped.
Outputs: results/geometry/adaptation/spectral_<dataset>_{lora_topk|allora}.csv and uniformity_<dataset>.csv
(columns uni_frozen, uni_adapted, d_uni).
"""
import argparse
import os
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from scipy.sparse.linalg import eigsh

from common import REPO, RESULTS, primary_condition
from converged_probe_gain import CK, EMB, stem

sys.path.insert(0, str(REPO))
from audio_comp.pipelines.rsa_cka_finetuned import CATEGORY_LOADERS  # noqa: E402


def metrics(G, k=10):
    s = np.sqrt(np.clip(np.diag(G), 1e-30, None))
    C = G / s[:, None] / s[None]
    iu = np.triu_indices(len(C), 1)
    out = dict(mean_energy=float(C.mean()),
               uniformity=float(np.log(np.mean(np.exp(-2.0 * np.clip(2.0 - 2.0 * C[iu], 0, None))))))
    r = C.mean(1)
    K = C - r[:, None] - r[None] + C.mean()
    tr = np.trace(K)
    out.update(pc1_share=eigsh(K, k=1, which="LA", return_eigenvectors=False)[0] / tr, part_ratio=tr ** 2 / np.sum(K * K))
    del K
    np.fill_diagonal(C, -np.inf)
    nn = np.argpartition(-C, k, axis=1)[:, :k]
    out["hub_cos"] = float(stats.skew(np.bincount(nn.ravel(), minlength=len(C))))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--cond", default="primary", choices=["primary", "allora"])
    a = ap.parse_args()
    clips = CATEGORY_LOADERS[a.dataset]()[0]
    frozen, rows = {}, []
    for ck in sorted(CK.glob(f"{a.dataset}_*_seed*.npz")):
        mm = re.match(rf"{a.dataset}_(lora|allora|topk)_(.+)_seed(\d)\.npz", ck.name)
        if not mm:
            continue
        cond, model, seed = mm.group(1), mm.group(2), int(mm.group(3))
        if cond != (primary_condition(model) if a.cond == "primary" else "allora"):
            continue
        if model not in frozen:
            ef = EMB / f"{a.dataset}__{model}.npz"
            if not ef.exists():
                frozen[model] = None
                continue
            z = np.load(ef, allow_pickle=True)
            lut = {stem(c): e for c, e in zip(z["clip_ids"], z["embeddings"])}
            Ef = np.stack([lut[stem(c["path"])] for c in clips if stem(c["path"]) in lut]).astype(np.float64)
            frozen[model] = metrics(Ef @ Ef.T)
        if frozen[model] is None:
            continue
        G = np.load(ck)["gram"].astype(np.float64)
        am = metrics(G)
        fm = frozen[model]
        if abs(am["mean_energy"] - fm["mean_energy"]) < 1e-4:
            print("adapted == frozen, skipping", ck.name, flush=True)
            continue
        rows.append(dict(dataset=a.dataset, model=model, condition=cond, seed=seed, n=len(G),
                         **{f"{k}_frozen": v for k, v in fm.items()}, **{f"{k}_adapted": v for k, v in am.items()},
                         **{f"d_{k}": am[k] - fm[k] for k in am}))
        print(rows[-1], flush=True)
    d = pd.DataFrame(rows)
    tag = "lora_topk" if a.cond == "primary" else "allora"
    out = RESULTS / "geometry" / "adaptation"
    d.drop(columns=[c for c in d.columns if "uniformity" in c]).to_csv(out / f"spectral_{a.dataset}_{tag}.csv", index=False)
    u = d[["dataset", "model", "condition", "seed", "n", "uniformity_frozen", "uniformity_adapted", "d_uniformity"]]
    u = u.rename(columns={"uniformity_frozen": "uni_frozen", "uniformity_adapted": "uni_adapted", "d_uniformity": "d_uni"})
    p = out / f"uniformity_{a.dataset}.csv"
    if p.exists():
        old = pd.read_csv(p)
        u = pd.concat([old[old.condition.isin(set(old.condition) - set(u.condition))], u])
    u.to_csv(p, index=False)


if __name__ == "__main__":
    main()
