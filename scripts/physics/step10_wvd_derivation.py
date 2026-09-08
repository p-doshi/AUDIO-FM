"""Step 10 - is PC1 the dominant mode of time-frequency energy concentration,
derivable WITHOUT the neural models?

For each probe clip we compute the Wigner-Ville second-moment tensor from
Cohen's global moments (no full WVD needed - the marginals are exact):

  z(t)   = analytic signal,  |z|^2 normalised to unit energy
  <t>    = sum t |z|^2                     sigma_t^2 = sum (t-<t>)^2 |z|^2
  <f>    = sum f |Z|^2                     sigma_f^2 = sum (f-<f>)^2 |Z|^2
  IF(t)  = (1/2pi) d/dt arg z(t)           (instantaneous frequency)
  cov_tf = sum t*IF(t)*|z|^2  -  <t><f>    (time-frequency covariance / chirp rate)

  T = [[sigma_t^2, cov_tf],
       [cov_tf,    sigma_f^2]]     -> eigenvalues l1>=l2, orientation theta,
                                      anisotropy (l1-l2)/(l1+l2)

sigma_f (bandwidth) is the continuous-space generalisation of the mel-Laplacian
term; IF / cov_tf generalise the group-delay term. So the WVD tensor is the
principled physics form of the two features PC1 loads on.

Tests:
  1. WVD-tensor RDM (z-scored Euclidean) vs the PC1-fitted RDM, vs group_delay
     & helmholtz RDMs, vs the 19-model consensus, vs each model.
  2. PCA of the per-clip WVD-tensor features -> its own PC1. RDM from projecting
     onto that 1-D axis, RSA vs the model-consensus PC1-fitted RDM.
     -> if high, PC1 = dominant TF-energy-concentration mode of THIS corpus,
        a statement independent of the models.

Outputs: results/physics_wvd_features.npz , results/physics_wvd_rsa.csv
"""
import time, numpy as np, pandas as pd, librosa
from scipy.signal import hilbert
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import squareform, pdist
import common as C

SR = 22050
MAXDUR = 10.0
EPS = 1e-12
COLS = ["log_sigma_t2", "log_sigma_f2", "cov_tf", "chirp_rate", "theta",
        "anisotropy", "mean_if_hz", "if_spread_hz", "loc_bw_hz_mean", "loc_bw_hz_std"]


def wvd_tensor(y, sr=SR):
    z = hilbert(y)
    p = np.abs(z) ** 2
    E = p.sum() + EPS
    t = np.arange(len(z)) / sr
    mt = (t * p).sum() / E
    st2 = ((t - mt) ** 2 * p).sum() / E
    Zf = np.fft.fft(z)
    ff = np.fft.fftfreq(len(z), 1 / sr)
    pos = ff >= 0
    pf = (np.abs(Zf) ** 2)[pos]
    f = ff[pos]
    Ef = pf.sum() + EPS
    mf = (f * pf).sum() / Ef
    sf2 = ((f - mf) ** 2 * pf).sum() / Ef
    phase = np.unwrap(np.angle(z))
    IF = np.gradient(phase) * sr / (2 * np.pi)            # Hz
    mif = (IF * p).sum() / E
    if_spread = np.sqrt(np.maximum(((IF - mif) ** 2 * p).sum() / E, 0))
    cov = (t * IF * p).sum() / E - mt * mif
    T = np.array([[st2, cov / sr], [cov / sr, sf2 / sr**2]])  # rescale for cond.
    w = np.linalg.eigvalsh(T)
    aniso = (w[-1] - w[0]) / (w[-1] + w[0] + EPS)
    theta = 0.5 * np.arctan2(2 * T[0, 1], T[0, 0] - T[1, 1])
    chirp = cov / (st2 + EPS)
    # local bandwidth per 1 s window (windowed pseudo-WVD proxy)
    w_s = int(sr)
    bws = []
    for s in range(0, max(1, len(y) - w_s + 1), w_s):
        seg = z[s:s + w_s]
        if len(seg) < w_s // 2:
            break
        _ff = np.fft.fftfreq(len(seg), 1 / sr); _pos = _ff >= 0
        Zs = (np.abs(np.fft.fft(seg)) ** 2)[_pos]
        fs = _ff[_pos]
        es = Zs.sum() + EPS
        mfs = (fs * Zs).sum() / es
        bws.append(np.sqrt(max(((fs - mfs) ** 2 * Zs).sum() / es, 0)))
    bws = np.array(bws) if bws else np.array([np.sqrt(sf2)])
    return np.array([np.log(st2 + EPS), np.log(sf2 + EPS), cov, chirp, theta,
                     aniso, mif, if_spread, bws.mean(), bws.std()])


def main():
    clips = C.select_clips()
    n = len(clips)
    print(f"[step10] WVD second-moment tensor for N={n} clips")
    feats, ids, cats = [], [], []
    t0 = time.time()
    for i, r in clips.iterrows():
        try:
            y, _ = librosa.load(r.abspath, sr=SR, mono=True, duration=MAXDUR)
            if np.any(y):
                y = librosa.util.normalize(y)
            feats.append(wvd_tensor(y))
            ids.append(r.clip_id); cats.append(r.category)
        except Exception as e:
            print(f"  ! {r.clip_id}: {e}")
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{n} ({time.time()-t0:.0f}s)")
    A = np.vstack(feats)
    for c in range(A.shape[1]):
        col = A[:, c]; bad = ~np.isfinite(col)
        if bad.any():
            col[bad] = np.nanmedian(col[~bad])
    np.savez(f"{C.OUT}/physics_wvd_features.npz",
             features=A, col_names=np.array(COLS), clip_ids=np.array(ids),
             categories=np.array(cats))
    print(f"[step10] features {A.shape}  cols={COLS}")

    # align to Step-1 clip order
    meta = np.load(f"{C.OUT}/rdm_tensor_meta.npz", allow_pickle=True)
    order = {c: i for i, c in enumerate(ids)}
    keep = [order[c] for c in meta["clip_ids"] if c in order]
    kept_ids = [c for c in meta["clip_ids"] if c in order]
    A = A[keep]
    Z = (A - A.mean(0)) / (A.std(0) + EPS)
    iu = np.triu_indices(len(kept_ids), 1)

    tensor = np.load(f"{C.OUT}/rdm_tensor.npy")
    co = list(np.load(f"{C.OUT}/physics_rdms/_clip_order.npy", allow_pickle=True))
    ck = np.array([co.index(c) for c in kept_ids])
    def grp(g):
        return rankdata(np.load(f"{C.OUT}/physics_rdms/{g}.npy")[np.ix_(ck, ck)][iu])
    Xg = np.column_stack([grp(g) for g in C.FEATURE_GROUPS])
    Xg = (Xg - Xg.mean(0)) / Xg.std(0)
    tk = np.array([list(meta["clip_ids"]).index(c) for c in kept_ids])
    mean19 = np.stack([rankdata(tensor[np.ix_(tk, tk, [k])][:, :, 0][iu])
                       for k in range(19)], 0).mean(0)
    pc1_fit = Xg @ (np.linalg.pinv(Xg.T @ Xg) @ (Xg.T @ ((mean19 - mean19.mean()) / mean19.std())))

    wvd_rdm = squareform(pdist(Z, metric="euclidean"))[iu]
    # WVD's own PC1
    u, sv, vt = np.linalg.svd(Z - Z.mean(0), full_matrices=False)
    wvd_pc1_score = (Z - Z.mean(0)) @ vt[0]
    wvd_pc1_rdm = np.abs(wvd_pc1_score[iu[0]] - wvd_pc1_score[iu[1]])
    wvd_ev = sv**2 / (sv**2).sum()

    rows = []
    def add(label, vec):
        rows.append(dict(comparison=label,
                         spearman=round(spearmanr(vec, wvd_rdm).statistic, 3),
                         spearman_wvdPC1only=round(spearmanr(vec, wvd_pc1_rdm).statistic, 3)))
    add("PC1-fitted (model consensus, physics part)", pc1_fit)
    add("mean-19-model consensus RDM", mean19)
    for g in C.FEATURE_GROUPS:
        add(f"physics group: {g}", grp(g))
    for k in range(19):
        add(f"model: {C.MODELS[k]}", rankdata(tensor[np.ix_(tk, tk, [k])][:, :, 0][iu]))
    df = pd.DataFrame(rows)
    df.to_csv(f"{C.OUT}/physics_wvd_rsa.csv", index=False)
    print(f"\n[step10] WVD-tensor variance: "
          + ", ".join(f"PC{i+1} {wvd_ev[i]*100:.0f}%" for i in range(4)))
    print("\n[step10] RSA of the WVD second-moment RDM (full tensor | WVD-PC1 only):")
    print(df.to_string(index=False))
    print(f"\n[step10] saved results/physics_wvd_features.npz, results/physics_wvd_rsa.csv")


if __name__ == "__main__":
    main()
