"""Step 4 - RSA of every model RDM against every physics feature-group RDM.

NO averaging across models. For each (model, feature_group) pair: Spearman
correlation on the upper triangle, with a 95% CI via the Fisher z-transform
(se = 1/sqrt(n_pairs - 3), n_pairs = number of clip pairs).

Saves results/physics_rsa_per_model.csv  (long format: model, feature_group,
spearman_rho, ci_low, ci_high, n_pairs).
"""
import numpy as np, pandas as pd
from scipy.stats import spearmanr
import common as C


def fisher_ci(rho, n, alpha=0.05):
    from scipy.stats import norm
    if abs(rho) >= 1 or n <= 3:
        return (np.nan, np.nan)
    z = np.arctanh(rho)
    se = 1.0 / np.sqrt(n - 3)
    zc = norm.ppf(1 - alpha / 2)
    return (np.tanh(z - zc * se), np.tanh(z + zc * se))


def main():
    tensor = np.load(f"{C.OUT}/rdm_tensor.npy")
    meta = np.load(f"{C.OUT}/rdm_tensor_meta.npz", allow_pickle=True)
    clip_order = np.load(f"{C.OUT}/physics_rdms/_clip_order.npy", allow_pickle=True)
    # indices into the tensor for clips that have physics features
    tpos = {c: i for i, c in enumerate(meta["clip_ids"])}
    keep = np.array([tpos[c] for c in clip_order])
    n = len(keep)
    iu = np.triu_indices(n, 1)
    npairs = len(iu[0])
    print(f"[step4] N={n} clips, {npairs} pairs, {len(C.MODELS)} models x "
          f"{len(C.FEATURE_GROUPS)} feature groups")

    groups = C.FEATURE_GROUPS + ["_ALL"]
    phys_ut = {}
    for g in groups:
        rdm = np.load(f"{C.OUT}/physics_rdms/{g}.npy")
        phys_ut[g] = rdm[iu]

    rows = []
    for k, mdl in enumerate(C.MODELS):
        mrdm = tensor[np.ix_(keep, keep, [k])][:, :, 0]
        m_ut = mrdm[iu]
        for g in groups:
            rho, p = spearmanr(m_ut, phys_ut[g])
            lo, hi = fisher_ci(rho, npairs)
            rows.append(dict(model=mdl, feature_group=g, spearman_rho=rho,
                             p_value=p, ci_low=lo, ci_high=hi, n_pairs=npairs))
    df = pd.DataFrame(rows)
    df.to_csv(f"{C.OUT}/physics_rsa_per_model.csv", index=False)

    piv = df[df.feature_group != "_ALL"].pivot(
        index="model", columns="feature_group", values="spearman_rho")
    piv = piv.reindex(C.MODELS)
    pd.set_option("display.width", 200, "display.max_columns", 20)
    print(piv.round(3))
    print("\n_ALL (all physics features combined):")
    print(df[df.feature_group == "_ALL"].set_index("model")["spearman_rho"]
          .reindex(C.MODELS).round(3))
    print(f"[step4] saved {C.OUT}/physics_rsa_per_model.csv")


if __name__ == "__main__":
    main()
