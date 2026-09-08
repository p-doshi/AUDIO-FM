"""Fixed, differentiable-friendly physics frontend motivated by the Step 7
PC1 axis (group delay profile, rho=0.71) + the helmholtz/spectral-Laplacian
term (rho=0.69), with an optional PC2 branch (IMF modal energy, rho=0.73).

PC1 as an equation, per STFT frame t:
    tau(omega,t)  = -d phi(omega,t) / d omega           (group delay)
    g16(t)        = octave-band average of tau over 16 log-spaced bands
    gskew(t)      = skew_omega( band-averaged tau )
    lap(m,t)      = (D2_t + D2_m) log |S_mel(m,t)|       (T-F Laplacian, "div-like")
    frontend(t)   = [ g16(t) (16) , gskew(t) (1) , lap(:,t) (n_mels) ]

The frame-level output is a sequence  (T, 17 + n_mels).  IMF energy (8) is a
clip-level side vector (EMD is global), optionally computed per 1 s window.

This module is pure numpy so it can run without torch; the torch model
(step8) reimplements the same ops on tensors for the learned head, but the
cached features here are what training actually consumes.
"""
import numpy as np, librosa
from scipy import signal, stats

SR = 22050
N_FFT = 1024
HOP = 512                       # T ~ 431 for a 10 s clip
N_MELS = 48
N_GD_BANDS = 16
MAX_DUR = 10.0
EPS = 1e-10
FRONTEND_DIM = N_GD_BANDS + 1 + N_MELS      # 65


def _octave_edges(fmin, fmax, n):
    return np.logspace(np.log10(fmin), np.log10(fmax), n + 1)


def group_delay_frames(y, sr=SR):
    """(T, 16) octave-band group delay + (T,) cross-band skew, per frame."""
    S = librosa.stft(y, n_fft=N_FFT, hop_length=HOP, window="hann")
    phase = np.unwrap(np.angle(S), axis=0)
    freqs = np.fft.rfftfreq(N_FFT, 1.0 / sr)
    domega = 2 * np.pi * (freqs[1] - freqs[0])
    gd = -np.gradient(phase, axis=0) / domega              # (F, T) seconds
    edges = _octave_edges(max(freqs[1], 20.0), freqs[-1], N_GD_BANDS)
    T = gd.shape[1]
    bands = np.zeros((T, N_GD_BANDS), np.float32)
    for b in range(N_GD_BANDS):
        m = (freqs >= edges[b]) & (freqs < edges[b + 1])
        bands[:, b] = gd[m].mean(0) if m.any() else 0.0
    # fill empty low bands with nearest populated
    filled = np.where(bands.any(0))[0]
    for b in range(N_GD_BANDS):
        if not bands[:, b].any() and len(filled):
            bands[:, b] = bands[:, filled[np.argmin(np.abs(filled - b))]]
    with np.errstate(all="ignore"):
        sk = stats.skew(bands, axis=1)
    # frames with ~flat group delay across bands -> undefined skew -> 0
    sk = np.nan_to_num(sk, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    return bands, sk


def reassigned_group_delay_profile(y, sr=SR, n_fft=None):
    """Group-delay profile via the REASSIGNED spectrogram (stable IF/GD from
    the ratio of two STFTs, not finite-differencing wrapped phase).

    Returns (16,) energy-weighted mean local group delay per octave band,
    plus the cross-band skew. `local group delay = t_reassigned - t_frame`.
    """
    n_fft = n_fft or N_FFT
    if len(y) < n_fft:
        y = np.pad(y, (0, n_fft - len(y)))
    freqs_r, times_r, mags = librosa.reassigned_spectrogram(
        y, sr=sr, n_fft=n_fft, hop_length=HOP, window="hann",
        center=True, fill_nan=True, clip=True)
    nfr = mags.shape[1]
    frame_t = librosa.frames_to_time(np.arange(nfr), sr=sr, hop_length=HOP,
                                     n_fft=n_fft)
    gd = times_r - frame_t[None, :]                    # (F, T) local group delay
    w = np.maximum(mags, 0.0)
    nyq = np.fft.rfftfreq(n_fft, 1.0 / sr)
    edges = _octave_edges(max(nyq[1], 20.0), nyq[-1], N_GD_BANDS)
    prof = np.zeros(N_GD_BANDS, np.float64)
    for b in range(N_GD_BANDS):
        m = (nyq >= edges[b]) & (nyq < edges[b + 1])
        if not m.any():
            continue
        gg = gd[m]; ww = w[m]
        sw = ww.sum()
        prof[b] = (gg * ww).sum() / sw if sw > 1e-12 else np.nanmean(gg)
    filled = np.where(prof != 0)[0]
    for b in range(N_GD_BANDS):
        if prof[b] == 0 and len(filled):
            prof[b] = prof[filled[np.argmin(np.abs(filled - b))]]
    with np.errstate(all="ignore"):
        sk = float(stats.skew(prof))
    if not np.isfinite(sk):
        sk = 0.0
    return np.nan_to_num(prof), sk


def spectral_laplacian_frames(y, sr=SR):
    """(T, n_mels) discrete T-F Laplacian of the log-mel spectrogram
    (the 'divergence-like' helmholtz-proxy term)."""
    M = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=N_FFT, hop_length=HOP, n_mels=N_MELS))   # (n_mels, T)
    d2m = np.gradient(np.gradient(M, axis=0), axis=0)
    d2t = np.gradient(np.gradient(M, axis=1), axis=1)
    lap = (d2m + d2t).T.astype(np.float32)                          # (T, n_mels)
    return lap


def imf_energy_windows(y, sr=SR, win_s=1.0, n_imf=8):
    """(n_win, 8) IMF energy fractions per 1 s window (PC2 branch).
    EMD on an 8 kHz decimation of each window."""
    from PyEMD import EMD
    tgt = 8000
    yd = signal.resample_poly(y, tgt, sr)
    w = int(tgt * win_s)
    out = []
    for s in range(0, max(1, len(yd) - w + 1), w):
        seg = yd[s:s + w]
        if len(seg) < w // 2:
            break
        emd = EMD(); emd.FIXE_H = 5
        try:
            imfs = emd.emd(seg.astype(np.float64), max_imf=n_imf)
            e = np.array([np.sum(c ** 2) for c in imfs])
            f = np.zeros(n_imf); f[:min(n_imf, len(e))] = (e / (e.sum() + EPS))[:n_imf]
        except Exception:
            f = np.zeros(n_imf)
        out.append(f)
    return np.array(out, np.float32) if out else np.zeros((1, n_imf), np.float32)


def frontend_frames(path, want_imf=False):
    y, _ = librosa.load(path, sr=SR, mono=True, duration=MAX_DUR)
    if np.any(y):
        y = librosa.util.normalize(y)
    g16, gsk = group_delay_frames(y)
    lap = spectral_laplacian_frames(y)
    T = min(g16.shape[0], lap.shape[0])
    feat = np.concatenate([g16[:T], gsk[:T, None], lap[:T]], axis=1)   # (T, 65)
    imf = imf_energy_windows(y) if want_imf else None
    return feat.astype(np.float32), imf
