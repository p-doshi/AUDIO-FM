"""Step 13 - controlled synthetic test of physical-parameter recovery
(direction 3a) + an empirical check of the covariance claim (direction 2).

Setup: a bank of source signals (broadband clicks, chirps, tonal + harmonic
stacks, filtered noise) is passed through a parametrised linear DISPERSIVE
channel with transfer function
     H(w; a,b,g,tau0) = exp( -i * phi(w) ) * exp( -g * (w/pi)^2 )
     phi(w) = tau0*w + a*w^2 + b*w^3        (group delay tau_g(w) = -dphi/dw
                                             = -tau0 - 2a*w - 3b*w^2)
so (a, b) set the linear and quadratic dispersion (the "c(w)" curve), g sets
frequency-dependent attenuation, tau0 a bulk delay. Random (a,b,g,tau0,source)
per example.

Q1 (param recovery): regress (a, b, g) from the PC1 physics frontend vs from
   log-mel (both mean-pooled, same ridge regressor). Lower error from the
   physics frontend => it exposes the propagation parameters more directly.
Q2 (covariance): the frontend's group-delay-profile channel should shift by
   exactly tau_g(w) when the channel is applied. Measure
   corr( delta(frontend group delay) , true tau_g(w) ) across examples.

Output: results/physics_dispersion_recovery.csv
"""
import numpy as np, pandas as pd
from numpy.fft import rfft, irfft, rfftfreq
from sklearn.linear_model import Ridge
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import r2_score, mean_absolute_error
import librosa
import phys_frontend as PF
import common as C

SR = PF.SR
DUR = 4.0
N = int(SR * DUR)
RNG = np.random.RandomState(0)
NEX = 900


def source(kind):
    t = np.arange(N) / SR
    if kind == "click":
        x = np.zeros(N)
        for _ in range(RNG.randint(2, 6)):
            x[RNG.randint(N)] = RNG.randn()
        x = np.convolve(x, np.hanning(64), "same")
    elif kind == "chirp":
        f0, f1 = RNG.uniform(80, 400), RNG.uniform(2000, 9000)
        x = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * DUR) * t**2))
    elif kind == "harmonic":
        f0 = RNG.uniform(90, 300)
        x = sum(np.sin(2 * np.pi * k * f0 * t) / k for k in range(1, RNG.randint(4, 12)))
    else:  # filtered noise
        x = np.random.randn(N)
        b = np.hanning(RNG.randint(20, 200)); x = np.convolve(x, b, "same")
    x += 0.02 * np.random.randn(N)
    return (x / (np.abs(x).max() + 1e-9)).astype(np.float64)


def apply_channel(x, a, b, g, tau0):
    X = rfft(x)
    w = 2 * np.pi * rfftfreq(N, 1 / SR)
    wn = w / w.max()
    phi = tau0 * w + a * w**2 + b * w**3
    H = np.exp(-1j * phi) * np.exp(-g * wn**2)
    return irfft(X * H, n=N)


def pooled_frontend(y):
    y = librosa.util.normalize(y.astype(np.float32))
    g16, gsk = PF.group_delay_frames(y)
    lap = PF.spectral_laplacian_frames(y)
    T = min(len(g16), len(lap))
    return (np.nan_to_num(np.concatenate([g16[:T].mean(0), [np.nan_to_num(gsk[:T]).mean()],
                                          lap[:T].mean(0)])),
            np.nan_to_num(g16[:T]).mean(0))          # also return the 16-band GD profile


def pooled_logmel(y):
    y = librosa.util.normalize(y.astype(np.float32))
    M = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=PF.N_FFT, hop_length=PF.HOP, n_mels=PF.N_MELS))
    return M.mean(1)


def true_group_delay_profile(a, b):
    """tau_g(w) = -tau0-2a w-3b w^2, sampled at the 16 octave band centres
    (tau0 drops out as a constant offset -> use the a,b part)."""
    freqs = rfftfreq(PF.N_FFT, 1 / SR)
    edges = np.logspace(np.log10(max(freqs[1], 20)), np.log10(freqs[-1]), PF.N_GD_BANDS + 1)
    fc = np.sqrt(edges[:-1] * edges[1:]); w = 2 * np.pi * fc
    return -(2 * a * w + 3 * b * w**2)


def main():
    kinds = ["click", "chirp", "harmonic", "noise"]
    Xf, Xm, Y, dGD, trueGD = [], [], [], [], []
    print(f"[step13] generating {NEX} dispersive-channel examples ...")
    for i in range(NEX):
        s = source(kinds[i % 4])
        a = RNG.uniform(-3e-8, 3e-8)      # quad dispersion
        b = RNG.uniform(-2e-12, 2e-12)    # cubic dispersion
        g = RNG.uniform(0.0, 3.0)         # freq-dependent attenuation
        tau0 = RNG.uniform(0, 0.02)
        y = apply_channel(s, a, b, g, tau0)
        f_full, gd_out = pooled_frontend(y)
        _, gd_in = pooled_frontend(s)
        Xf.append(f_full); Xm.append(pooled_logmel(y))
        Y.append([a * 1e8, b * 1e12, g])
        dGD.append(gd_out - gd_in); trueGD.append(true_group_delay_profile(a, b))
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{NEX}")
    Xf, Xm, Y = np.array(Xf), np.array(Xm), np.array(Y)
    dGD, trueGD = np.array(dGD), np.array(trueGD)

    # ---- Q1: parameter recovery ----
    rows = []
    for label, Xr in [("PC1_frontend", Xf), ("logmel", Xm)]:
        Xz = (Xr - Xr.mean(0)) / (Xr.std(0) + 1e-9)
        for j, pname in enumerate(["a_quad_disp", "b_cubic_disp", "g_attenuation"]):
            pred = cross_val_predict(Ridge(alpha=1.0), Xz, Y[:, j], cv=5)
            rows.append(dict(representation=label, parameter=pname,
                             r2=round(r2_score(Y[:, j], pred), 3),
                             mae=round(mean_absolute_error(Y[:, j], pred), 3)))
    df = pd.DataFrame(rows)

    # ---- Q2: covariance of the group-delay channel ----
    # per-example correlation between measured frontend GD shift and true tau_g(w)
    cc = np.array([np.corrcoef(dGD[i] - dGD[i].mean(), trueGD[i] - trueGD[i].mean())[0, 1]
                   for i in range(NEX) if np.std(trueGD[i]) > 1e-30])
    cov_row = dict(representation="PC1_frontend", parameter="COVARIANCE_check",
                   r2=round(float(np.nanmean(cc)), 3),
                   mae=round(float(np.nanmedian(cc)), 3))
    df = pd.concat([df, pd.DataFrame([cov_row])], ignore_index=True)
    df.to_csv(f"{C.OUT}/physics_dispersion_recovery.csv", index=False)

    print("\n[step13] Q1 - recover dispersion/attenuation params (5-fold CV R^2):")
    print(df[df.parameter != "COVARIANCE_check"].pivot(
        index="parameter", columns="representation", values="r2").to_string())
    print(f"\n[step13] Q2 - covariance: mean corr(frontend group-delay shift, "
          f"true tau_g(w)) = {np.nanmean(cc):.3f}  (median {np.nanmedian(cc):.3f}, "
          f"n={len(cc)})")
    print(f"\n[step13] saved {C.OUT}/physics_dispersion_recovery.csv")


if __name__ == "__main__":
    main()
