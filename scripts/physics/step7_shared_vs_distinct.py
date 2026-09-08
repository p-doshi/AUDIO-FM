"""Step 7 - is the ~37% physical substrate the SAME across models, or do
different models use different physical dimensions within that ceiling?

Method
------
Pair-level predictors = the 8 physics feature-group RDMs (upper triangle),
rank-transformed. For each model:
  * multiple linear regression of rank(model RDM) on the 8 rank(group RDM)s
    -> standardized betas (physical "loading" vector) + R^2 (how much of the
       model's geometry the physics basis explains)
  * fitted RDM   = physics-explainable part of that model's geometry
  * residual RDM = the rest

Then:
  A. 19x8 standardized-beta matrix -> 19x19 loading-similarity matrix
     (do models weight the physical dimensions the same way?)
  B. 19x19 RSA of the FITTED RDMs vs 19x19 RSA of the RAW RDMs vs the
     RESIDUAL RDMs (is cross-model agreement higher, equal or lower once
     you restrict to the physics-explainable part?)
  C. PCA of the 19 fitted-RDM vectors -> how many independent physical axes
     does the roster actually use, and where does each model load?

Outputs:
  results/physics_shared_betas.csv          (19 x 8 standardized betas + R2)
  results/physics_shared_fitted_rsa.csv     (19x19 RSA of fitted RDMs)
  results/physics_shared_raw_rsa.csv        (19x19 RSA of raw RDMs)
  results/physics_shared_resid_rsa.csv      (19x19 RSA of residual RDMs)
  results/physics_shared_pca.csv            (model loadings on fitted-RDM PCs)
  console: summary tables
"""
import numpy as np, pandas as pd
from scipy.stats import rankdata, spearmanr
import common as C

pd.set_option("display.width", 240, "display.max_columns", 40)
GROUPS = C.FEATURE_GROUPS  # 8


def z(v):
    v = np.asarray(v, float)
    return (v - v.mean()) / (v.std() + 1e-12)


def zrank(v):
    return z(rankdata(v))


def main():
    tensor = np.load(f"{C.OUT}/rdm_tensor.npy")
    meta = np.load(f"{C.OUT}/rdm_tensor_meta.npz", allow_pickle=True)
    clip_order = np.load(f"{C.OUT}/physics_rdms/_clip_order.npy", allow_pickle=True)
    tpos = {c: i for i, c in enumerate(meta["clip_ids"])}
    keep = np.array([tpos[c] for c in clip_order])
    n = len(keep)
    iu = np.triu_indices(n, 1)
    npairs = len(iu[0])
    print(f"[step7] N={n} clips, {npairs} pairs, {len(C.MODELS)} models, "
          f"{len(GROUPS)} physics dimensions")

    # design matrix: 8 rank-z physics group RDMs
    X = np.column_stack([zrank(np.load(f"{C.OUT}/physics_rdms/{g}.npy")[iu])
                         for g in GROUPS])
    XtX_inv = np.linalg.pinv(X.T @ X)
    # collinearity of the physics basis
    cc = np.corrcoef(X.T)
    print("\n[step7] physics-dimension collinearity (max |r| off-diag = "
          f"{np.abs(cc - np.eye(8)).max():.2f})")

    betas, r2s, fitted, resid = {}, {}, {}, {}
    raw = {}
    for k, mdl in enumerate(C.MODELS):
        y = zrank(tensor[np.ix_(keep, keep, [k])][:, :, 0][iu])
        raw[mdl] = y
        b = XtX_inv @ (X.T @ y)
        yhat = X @ b
        betas[mdl] = b
        r2s[mdl] = 1 - np.var(y - yhat) / np.var(y)
        fitted[mdl] = yhat
        resid[mdl] = y - yhat

    B = pd.DataFrame(betas, index=GROUPS).T[GROUPS]
    B["R2"] = pd.Series(r2s)
    B = B.loc[C.MODELS]
    B.round(3).to_csv(f"{C.OUT}/physics_shared_betas.csv")
    print("\n[step7] standardized betas (physical loading vector) + R^2 per model")
    print(B.round(3).to_string())

    # ---- A. loading-vector similarity across models ----
    Bmat = B[GROUPS].values
    Lsim = np.corrcoef(Bmat)
    Ldf = pd.DataFrame(Lsim, index=C.MODELS, columns=C.MODELS)
    print("\n[step7] A. cross-model similarity of the 8-D physical loading vector")
    print(f"   mean off-diagonal r = {Lsim[np.triu_indices(19,1)].mean():.3f}")
    print(f"   min pair r = {Lsim[np.triu_indices(19,1)].min():.3f}  "
          f"max pair r = {Lsim[np.triu_indices(19,1)].max():.3f}")

    # ---- B. RSA of raw vs fitted vs residual, across models ----
    def rsa_matrix(d):
        M = np.zeros((19, 19))
        keys = C.MODELS
        for i in range(19):
            for j in range(i, 19):
                r = spearmanr(d[keys[i]], d[keys[j]]).statistic
                M[i, j] = M[j, i] = r
        return pd.DataFrame(M, index=keys, columns=keys)

    raw_rsa = rsa_matrix(raw)
    fit_rsa = rsa_matrix(fitted)
    res_rsa = rsa_matrix(resid)
    raw_rsa.to_csv(f"{C.OUT}/physics_shared_raw_rsa.csv")
    fit_rsa.to_csv(f"{C.OUT}/physics_shared_fitted_rsa.csv")
    res_rsa.to_csv(f"{C.OUT}/physics_shared_resid_rsa.csv")
    ut = np.triu_indices(19, 1)
    print("\n[step7] B. cross-model RSA: mean +/- sd of off-diagonal pairs")
    print(f"   RAW model RDMs      : {raw_rsa.values[ut].mean():.3f} +/- {raw_rsa.values[ut].std():.3f}")
    print(f"   FITTED (physics)    : {fit_rsa.values[ut].mean():.3f} +/- {fit_rsa.values[ut].std():.3f}")
    print(f"   RESIDUAL (non-phys) : {res_rsa.values[ut].mean():.3f} +/- {res_rsa.values[ut].std():.3f}")
    print("   -> fitted >> raw  => physical substrate is largely SHARED/convergent")
    print("   -> fitted ~  raw  => models use DIFFERENT physical dimensions")

    # ---- C. PCA of the 19 fitted-RDM vectors ----
    F = np.column_stack([z(fitted[m]) for m in C.MODELS])  # (npairs, 19)
    # SVD on the pair x model matrix; components live in model space
    U, S, Vt = np.linalg.svd(F - F.mean(0), full_matrices=False)
    ev = S**2 / np.sum(S**2)
    print("\n[step7] C. PCA of the 19 fitted (physics-explainable) RDMs")
    print("   variance explained:",
          ", ".join(f"PC{i+1} {ev[i]*100:.1f}%" for i in range(6)))
    print(f"   PC1 alone = {ev[0]*100:.1f}%  |  PC1-3 = {ev[:3].sum()*100:.1f}%")
    load = pd.DataFrame(Vt[:5].T, index=C.MODELS,
                        columns=[f"PC{i+1}" for i in range(5)])
    load["var_expl_by_PC1"] = (Vt[0].T**2)  # share of each model on PC1 direction
    load.to_csv(f"{C.OUT}/physics_shared_pca.csv")
    print("\n   model loadings on fitted-RDM principal axes:")
    print(load[[f"PC{i+1}" for i in range(4)]].round(3).to_string())

    # which physics group each fitted-PC corresponds to
    print("\n   correlation of each fitted-RDM PC with the 8 physics-group RDMs:")
    pc_pair = (F - F.mean(0)) @ Vt.T  # (npairs, 19) scores
    rows = []
    for p in range(4):
        rows.append({g: spearmanr(pc_pair[:, p], X[:, gi]).statistic
                     for gi, g in enumerate(GROUPS)})
    print(pd.DataFrame(rows, index=[f"PC{i+1}" for i in range(4)]).round(2).to_string())

    print("\n[step7] saved results/physics_shared_*.csv")


if __name__ == "__main__":
    main()
