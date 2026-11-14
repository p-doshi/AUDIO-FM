"""Step 14 - the gate: does a REASSIGNED-spectrogram group-delay estimator
exhibit dispersive covariance where the raw STFT-phase-derivative one failed
(Step 13), and does it recover the PC1 connection the noisy estimator missed?

Part A (synthetic, same 900-channel bank as Step 13):
  Q1  ridge-CV R^2 recovering (a, b, g) from the reassigned GD profile
  Q2  per-example corr( g_out(w) - g_in(w) , true tau_g(w) = -(2a w + 3b w^2) )
  DECISION: R^2(a) or R^2(b) > 0.4  ->  covariance back on the table.
            still ~0  ->  signal-model problem, not an estimator problem;
            real audio doesn't carry recoverable single-clip propagation info.

Part B (real probe clips, ~600):
  reassigned GD-profile RDM  vs  PC1-fitted RDM  vs  old STFT group_delay RDM
  vs  mean-19 consensus.  Does the better estimator raise RSA-with-PC1?

Output: results/physics_reassignment_gate.csv
"""
import time, numpy as np, pandas as pd, librosa
from numpy.fft import rfft, irfft, rfftfreq
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import squareform, pdist
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score
import phys_frontend as PF
import common as C
from step13_dispersion_recovery import source, apply_channel, SR, N, DUR, RNG
NEX = 500


def true_tau_g(a, b):
    nyq = rfftfreq(PF.N_FFT, 1 / SR)
    edges = np.logspace(np.log10(max(nyq[1], 20)), np.log10(nyq[-1]), PF.N_GD_BANDS + 1)
    fc = np.sqrt(edges[:-1] * edges[1:]); w = 2 * np.pi * fc
    return -(2 * a * w + 3 * b * w ** 2)


def part_a():
    print(f"[step14-A] {NEX} synthetic dispersive channels, reassigned GD estimator")
    kinds = ["click", "chirp", "harmonic", "noise"]
    Gout, Delta, TrueTG, Y = [], [], [], []
    t0 = time.time()
    for i in range(NEX):
        s = source(kinds[i % 4])
        a = RNG.uniform(-3e-8, 3e-8); b = RNG.uniform(-2e-12, 2e-12)
        g = RNG.uniform(0.0, 3.0); tau0 = RNG.uniform(0, 0.02)
        y = apply_channel(s, a, b, g, tau0)
        g_out, _ = PF.reassigned_group_delay_profile(y)
        g_in, _ = PF.reassigned_group_delay_profile(s)
        Gout.append(g_out); Delta.append(g_out - g_in)
        TrueTG.append(true_tau_g(a, b)); Y.append([a * 1e8, b * 1e12, g])
        if (i + 1) % 200 == 0:
            print(f"   {i+1}/{NEX} ({time.time()-t0:.0f}s)")
    Gout, Delta, TrueTG, Y = map(np.array, (Gout, Delta, TrueTG, Y))

    rows = []
    Xz = (Gout - Gout.mean(0)) / (Gout.std(0) + 1e-9)
    for j, p in enumerate(["a_quad_disp", "b_cubic_disp", "g_attenuation"]):
        pred = cross_val_predict(Ridge(alpha=1.0), Xz, Y[:, j], cv=5)
        rows.append(dict(part="A_synthetic", metric=f"R2_recover_{p}",
                         value=round(r2_score(Y[:, j], pred), 3)))
    cc = np.array([np.corrcoef(Delta[i] - Delta[i].mean(),
                               TrueTG[i] - TrueTG[i].mean())[0, 1]
                   for i in range(NEX) if np.std(TrueTG[i]) > 1e-30])
    rows.append(dict(part="A_synthetic", metric="covariance_corr_mean",
                     value=round(float(np.nanmean(cc)), 3)))
    rows.append(dict(part="A_synthetic", metric="covariance_corr_median",
                     value=round(float(np.nanmedian(cc)), 3)))
    r2a = [r["value"] for r in rows if r["metric"] == "R2_recover_a_quad_disp"][0]
    r2b = [r["value"] for r in rows if r["metric"] == "R2_recover_b_cubic_disp"][0]
    verdict = ("COVARIANCE RECOVERED - physics interpretation back on the table"
               if max(r2a, r2b) > 0.4 else
               "still ~0 - signal-model problem, not estimator; physics interp "
               "not rescuable at single-clip level")
    rows.append(dict(part="A_synthetic", metric="VERDICT", value=verdict))
    print(f"[step14-A] R2(a)={r2a}  R2(b)={r2b}  R2(g)="
          f"{[r['value'] for r in rows if 'g_att' in r['metric']][0]}  "
          f"covariance corr={np.nanmean(cc):.3f}")
    print(f"[step14-A] VERDICT: {verdict}")
    return rows


def part_b(n_clips=600):
    print(f"\n[step14-B] reassigned GD-profile RDM on {n_clips} real probe clips")
    meta = np.load(f"{C.OUT}/rdm_tensor_meta.npz", allow_pickle=True)
    ids_all = list(meta["clip_ids"])
    m = pd.read_csv(C.MANIFEST).set_index("clip_id")
    import os
    rng = np.random.RandomState(0)
    pick = np.sort(rng.choice(len(ids_all), min(n_clips, len(ids_all)), replace=False))
    ids = [ids_all[i] for i in pick]
    prof = []
    t0 = time.time()
    for k, cid in enumerate(ids):
        y, _ = librosa.load(os.path.join(C.DATA_ROOT, m.loc[cid, "path"]),
                            sr=SR, mono=True, duration=10.0)
        if np.any(y):
            y = librosa.util.normalize(y)
        p, sk = PF.reassigned_group_delay_profile(y)
        prof.append(np.concatenate([p, [sk]]))
        if (k + 1) % 200 == 0:
            print(f"   {k+1}/{len(ids)} ({time.time()-t0:.0f}s)")
    P = np.array(prof)
    Z = (P - P.mean(0)) / (P.std(0) + 1e-9)
    iu = np.triu_indices(len(ids), 1)
    reass_rdm = rankdata(squareform(pdist(Z, metric="euclidean"))[iu])

    tensor = np.load(f"{C.OUT}/rdm_tensor.npy")
    tp = {c: i for i, c in enumerate(ids_all)}
    tk = np.array([tp[c] for c in ids])
    old_gd = rankdata(np.load(f"{C.OUT}/physics_rdms/group_delay.npy")[np.ix_(pick, pick)][iu]) \
        if list(np.load(f"{C.OUT}/physics_rdms/_clip_order.npy", allow_pickle=True)) == ids_all else None
    co = list(np.load(f"{C.OUT}/physics_rdms/_clip_order.npy", allow_pickle=True))
    ck = np.array([co.index(c) for c in ids])
    old_gd = rankdata(np.load(f"{C.OUT}/physics_rdms/group_delay.npy")[np.ix_(ck, ck)][iu])
    Xg = np.column_stack([rankdata(np.load(f"{C.OUT}/physics_rdms/{g}.npy")[np.ix_(ck, ck)][iu])
                          for g in C.FEATURE_GROUPS])
    Xg = (Xg - Xg.mean(0)) / Xg.std(0)
    mean19 = np.stack([rankdata(tensor[np.ix_(tk, tk, [q])][:, :, 0][iu]) for q in range(19)], 0).mean(0)
    pc1_fit = Xg @ (np.linalg.pinv(Xg.T @ Xg) @ (Xg.T @ ((mean19 - mean19.mean()) / mean19.std())))

    rows = [
        dict(part="B_real", metric="reassignedGD_RDM vs PC1_fitted",
             value=round(spearmanr(reass_rdm, pc1_fit).statistic, 3)),
        dict(part="B_real", metric="reassignedGD_RDM vs mean19_consensus",
             value=round(spearmanr(reass_rdm, mean19).statistic, 3)),
        dict(part="B_real", metric="OLD_stft_GD_RDM vs PC1_fitted (ref)",
             value=round(spearmanr(old_gd, pc1_fit).statistic, 3)),
        dict(part="B_real", metric="reassignedGD_RDM vs OLD_stft_GD_RDM",
             value=round(spearmanr(reass_rdm, old_gd).statistic, 3)),
    ]
    for q in range(19):
        rows.append(dict(part="B_real", metric=f"reassignedGD_RDM vs model:{C.MODELS[q]}",
                         value=round(spearmanr(reass_rdm, rankdata(tensor[np.ix_(tk, tk, [q])][:, :, 0][iu])).statistic, 3)))
    for r in rows[:4]:
        print(f"   {r['metric']:42s} {r['value']:+.3f}")
    return rows


def main():
    rows = part_a() + part_b()
    pd.DataFrame(rows).to_csv(f"{C.OUT}/physics_reassignment_gate.csv", index=False)
    print(f"\n[step14] saved {C.OUT}/physics_reassignment_gate.csv")


if __name__ == "__main__":
    main()
