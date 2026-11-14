"""Step 2 - physics-grounded features from raw waveforms.

For every subsampled probe clip we compute 8 feature groups (see FEATURE_GROUPS).
Audio is loaded mono at SR Hz, first MAX_DUR_SEC seconds only.

Output: results/physics_features.npz
  - one 2-D array per feature group, shape (N, d_group), rows aligned to
    results/rdm_tensor_meta.npz clip order
  - clip_ids, categories, and per-group column names
  - `skipped` : list of clip_ids dropped (too short / unreadable)

Everything implemented with numpy/scipy/librosa/PyEMD. Higuchi is implemented
directly below (NOT antropy).
"""
import sys, time, warnings, numpy as np, librosa
from scipy import signal, stats
from PyEMD import EMD
import common as C

warnings.filterwarnings("ignore")
EPS = 1e-12
N_GD_BANDS = 16


# ----------------------------------------------------------------------------
def _octave_edges(fmin, fmax, n):
    return np.logspace(np.log10(fmin), np.log10(fmax), n + 1)


def group_delay_profile(y, sr):
    """STFT phase -> group delay (-dphi/domega) per frequency bin, averaged
    over time, then binned into 16 log/octave-spaced bands.
    Returns (16 band means, skew across the 16 bands)."""
    n_fft = 1024
    S = librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 4, window="hann")
    phase = np.unwrap(np.angle(S), axis=0)          # unwrap along frequency
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)
    domega = 2 * np.pi * (freqs[1] - freqs[0])
    # dphi/domega via central difference along frequency axis
    dphi = np.gradient(phase, axis=0) / domega
    gd = -dphi.mean(axis=1)                          # per frequency bin, sec
    edges = _octave_edges(max(freqs[1], 20.0), freqs[-1], N_GD_BANDS)
    band_means = np.full(N_GD_BANDS, np.nan)
    for b in range(N_GD_BANDS):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if m.any():
            band_means[b] = gd[m].mean()
    # fill any empty low band with nearest
    if np.isnan(band_means).any():
        idx = np.where(~np.isnan(band_means))[0]
        for b in np.where(np.isnan(band_means))[0]:
            band_means[b] = band_means[idx[np.argmin(np.abs(idx - b))]]
    sk = stats.skew(band_means)
    return band_means, float(sk), edges


def dispersion_fit(band_means, edges):
    """2nd-order polynomial fit of group delay vs log10(band centre freq).
    Returns (slope=linear coef, curvature=quadratic coef)."""
    fc = np.sqrt(edges[:-1] * edges[1:])
    x = np.log10(fc)
    coef = np.polyfit(x, band_means, 2)   # [quad, lin, const]
    return float(coef[1]), float(coef[0])


EMD_SR = 8000   # EMD is O(N * n_sift); decimate to bound cost. Modal energy
                # distribution is preserved well below the original band edges.


def imf_energy_fraction(y, sr, n_imf=8):
    """EMD decomposition -> fraction of total energy in each of first 8 IMFs.
    Signal decimated to EMD_SR first (speed; changes absolute IMF freqs but
    not the relative modal energy split we use)."""
    yd = signal.resample_poly(y, EMD_SR, sr) if sr != EMD_SR else y
    emd = EMD()
    emd.FIXE_H = 5          # cap sifting iterations for speed/determinism
    try:
        imfs = emd.emd(yd.astype(np.float64), max_imf=n_imf)
    except Exception:
        return np.zeros(n_imf)
    e = np.array([np.sum(c ** 2) for c in imfs])
    tot = e.sum() + EPS
    frac = np.zeros(n_imf)
    frac[:min(n_imf, len(e))] = (e / tot)[:n_imf]
    return frac


def higuchi_fd(y, kmax=8):
    """Higuchi fractal dimension - direct implementation.
    L(k) computed for k=1..kmax; FD = slope of log(L(k)) vs log(1/k)."""
    x = np.asarray(y, dtype=np.float64)
    N = len(x)
    Lk = []
    ks = []
    for k in range(1, kmax + 1):
        Lm = []
        for m in range(k):
            idx = np.arange(m, N, k)
            if len(idx) < 2:
                continue
            diff = np.abs(np.diff(x[idx])).sum()
            norm = (N - 1) / (len(idx) - 1) / k
            Lm.append(diff * norm / k)
        if Lm:
            Lk.append(np.mean(Lm))
            ks.append(k)
    Lk = np.array(Lk); ks = np.array(ks)
    slope = np.polyfit(np.log(1.0 / ks), np.log(Lk + EPS), 1)[0]
    return float(slope)


def coherence_decay_tau(y, sr, win_ms=50.0, hop_ms=10.0):
    """Short-time autocorrelation; fit exponential decay exp(-lag/tau) to the
    (frame-averaged) ACF envelope. Returns tau in seconds."""
    w = int(sr * win_ms / 1000)
    h = int(sr * hop_ms / 1000)
    if len(y) < w + h:
        return np.nan
    win = np.hanning(w)
    acfs = []
    for start in range(0, len(y) - w, h):
        fr = y[start:start + w] * win
        ac = librosa.autocorrelate(fr)              # FFT-based, length w
        ac = ac / (ac[0] + EPS)
        acfs.append(ac)
    acf = np.abs(np.mean(acfs, axis=0))
    env = np.maximum.accumulate(acf[::-1])[::-1]      # upper envelope
    lags = np.arange(len(env)) / sr
    mask = env > 1e-3
    if mask.sum() < 3:
        return np.nan
    slope = np.polyfit(lags[mask], np.log(env[mask]), 1)[0]
    tau = -1.0 / slope if slope < 0 else np.nan
    return float(tau)


def radiation_proxy(y, sr):
    """Octave-band energy weighted by f^2 (compact-source radiation efficiency).
    Returns (weighted/unweighted energy ratio, frequency centroid of the
    f^2-weighted band-energy distribution in Hz)."""
    n_fft = 2048
    S = np.abs(librosa.stft(y, n_fft=n_fft, hop_length=n_fft // 2)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = _octave_edges(max(freqs[1], 20.0), freqs[-1], 12)
    be, fc = [], []
    for b in range(len(edges) - 1):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        if m.any():
            be.append(S[m].sum())
            fc.append(np.sqrt(edges[b] * edges[b + 1]))
    be = np.array(be); fc = np.array(fc)
    w = be * fc ** 2
    ratio = w.sum() / (be.sum() + EPS)
    centroid = (fc * w).sum() / (w.sum() + EPS)
    # normalise ratio by fc^2 scale so it's O(1)-ish -> report log10
    return float(np.log10(ratio + EPS)), float(centroid)


def harmonic_features(y, sr):
    """HPSS harmonic/percussive energy ratio; if harmonic energy > 20% of
    total, estimate f0 and harmonic-series spacing from the ACF peak.
    Returns (harm/(harm+perc) fraction, f0_hz, harmonic_spacing_hz)."""
    h, p = librosa.effects.hpss(y)
    eh, ep = np.sum(h ** 2), np.sum(p ** 2)
    frac = eh / (eh + ep + EPS)
    f0, spacing = 0.0, 0.0
    if eh > 0.20 * (eh + ep + EPS):
        lo, hi = int(sr / 2000), int(sr / 50)       # 50 Hz .. 2000 Hz
        ac = librosa.autocorrelate(h, max_size=hi + 2)
        seg = ac[lo:hi]
        if len(seg) and seg.max() > 0:
            lag = lo + int(np.argmax(seg))
            f0 = sr / lag
            spacing = f0                            # harmonic spacing == f0
    return float(frac), float(f0), float(spacing)


def helmholtz_proxy(y, sr):
    """APPROXIMATE. Treat the (log) mel spectrogram as a scalar field on the
    time-frequency plane and take its 2-D gradient. This is NOT a true
    Helmholtz decomposition of the pressure field - there is no vector field
    here, only a scalar image. We form a pseudo-vector field from the gradient
    (gx, gy) and report proxies for its rotational vs divergent content:
      curl-like  ~ |d(gy)/dx - d(gx)/dy|   (should be ~0 for a pure gradient,
                   nonzero only from discretisation / cross structure)
      div-like   ~ |d(gx)/dx + d(gy)/dy|   (the Laplacian magnitude)
    Returned as (mean curl-like magnitude, mean div-like magnitude).
    """
    M = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=64, n_fft=1024, hop_length=256))
    gy, gx = np.gradient(M)                          # freq-axis, time-axis
    # Second derivatives via FORWARD differences (np.diff), not np.gradient:
    # under a central-difference stencil the discrete mixed partials are
    # identically equal so the curl is exactly 0 and carries no information.
    # Forward differencing makes the discrete mixed partials differ, and the
    # residual (curl_like) then picks up diagonal time-frequency structure
    # such as chirps / frequency sweeps, which is the phenomenon of interest.
    dgy_dx = np.diff(gy, axis=1)[:-1, :]
    dgx_dy = np.diff(gx, axis=0)[:, :-1]
    dgx_dx = np.diff(gx, axis=1)[:-1, :]
    dgy_dy = np.diff(gy, axis=0)[:, :-1]
    curl_like = np.abs(dgy_dx - dgx_dy)
    div_like = np.abs(dgx_dx + dgy_dy)
    return float(curl_like.mean()), float(div_like.mean())


# ----------------------------------------------------------------------------
def features_for_clip(path):
    y, _ = librosa.load(path, sr=C.SR, mono=True, duration=C.MAX_DUR_SEC)
    y = librosa.util.normalize(y) if np.any(y) else y
    if len(y) < int(C.SR * C.MIN_DUR_SEC):
        return None
    bm, sk, edges = group_delay_profile(y, C.SR)
    slope, curv = dispersion_fit(bm, edges)
    imf = imf_energy_fraction(y, C.SR)
    fd = higuchi_fd(y)
    tau = coherence_decay_tau(y, C.SR)
    rad_ratio, rad_centroid = radiation_proxy(y, C.SR)
    hfrac, f0, spacing = harmonic_features(y, C.SR)
    curl_like, div_like = helmholtz_proxy(y, C.SR)
    return dict(
        group_delay=np.concatenate([bm, [sk]]),
        dispersion_fit=np.array([slope, curv]),
        imf_energy=imf,
        fractal=np.array([fd]),
        coherence_decay=np.array([tau]),
        radiation=np.array([rad_ratio, rad_centroid]),
        harmonic=np.array([hfrac, f0, spacing]),
        helmholtz=np.array([curl_like, div_like]),
    )


COLS = {
    "group_delay": [f"gd_band{i:02d}" for i in range(N_GD_BANDS)] + ["gd_skew"],
    "dispersion_fit": ["disp_slope", "disp_curvature"],
    "imf_energy": [f"imf{i+1}_efrac" for i in range(8)],
    "fractal": ["higuchi_fd"],
    "coherence_decay": ["coherence_tau_s"],
    "radiation": ["radiation_logratio", "radiation_weighted_centroid_hz"],
    "harmonic": ["harmonic_frac", "f0_hz", "harmonic_spacing_hz"],
    "helmholtz": ["helmholtz_curl_like", "helmholtz_div_like"],
}


def main():
    clips = C.select_clips()
    n = len(clips)
    print(f"[step2] extracting physics features for N={n} clips "
          f"(sr={C.SR}, max_dur={C.MAX_DUR_SEC}s)")
    acc = {g: [] for g in C.FEATURE_GROUPS}
    kept_ids, kept_cat, skipped = [], [], []
    t0 = time.time()
    for i, r in clips.iterrows():
        try:
            f = features_for_clip(r.abspath)
        except Exception as e:
            f = None
            print(f"  ! {r.clip_id}: {type(e).__name__}: {e}")
        if f is None:
            skipped.append(r.clip_id)
            continue
        for g in C.FEATURE_GROUPS:
            acc[g].append(f[g])
        kept_ids.append(r.clip_id); kept_cat.append(r.category)
        if (len(kept_ids)) % 100 == 0:
            dt = time.time() - t0
            print(f"  {len(kept_ids)}/{n}  ({dt:.0f}s, {dt/len(kept_ids):.2f}s/clip)")
    out = {g: np.vstack(acc[g]) for g in C.FEATURE_GROUPS}
    for g in C.FEATURE_GROUPS:
        # replace non-finite with column median
        A = out[g]
        for c in range(A.shape[1]):
            col = A[:, c]
            bad = ~np.isfinite(col)
            if bad.any():
                col[bad] = np.nanmedian(col[~bad]) if (~bad).any() else 0.0
        out[g] = A
        print(f"  group {g:16s} shape={A.shape}  cols={COLS[g]}")
    np.savez(f"{C.OUT}/physics_features.npz",
             clip_ids=np.array(kept_ids), categories=np.array(kept_cat),
             skipped=np.array(skipped, dtype=object),
             col_names=np.array([f"{g}:{c}" for g in C.FEATURE_GROUPS for c in COLS[g]], dtype=object),
             **out)
    print(f"[step2] kept={len(kept_ids)} skipped={len(skipped)} "
          f"({skipped[:10]}{'...' if len(skipped)>10 else ''})")
    print(f"[step2] saved {C.OUT}/physics_features.npz  total {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
