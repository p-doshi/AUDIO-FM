# ============================================================================
# READ-ONLY REFERENCE COPY.  Canonical runnable location:
#     data/AIS/physics_vessel_transfer.py   (gitignored — confidential data area)
#
# This copy exists so the Step-11 logic is visible on the echo-physics-investigation
# branch. It will NOT run from scripts/physics/ — it imports data/AIS modules
# (remote_stream, train_vessel_classifier_streaming, frozen_baseline_streaming).
# See scripts/physics/README.md section 5. Run the canonical copy from data/AIS/,
# yourself, never via the assistant — same confidentiality rules as every data/AIS script.
# ============================================================================

"""OOD transfer test on the confidential vessel-acoustic data: does the
wave-physics representational substrate (Step 7's PC1 axis) generalise to a
domain no model in this project was trained on?

*** RUN THIS YOURSELF, IN YOUR OWN TERMINAL. ***  Same confidentiality rules
as every other data/AIS script:
  - streams vessel audio via RemoteStream (one bigdata6 password prompt),
    caches waveforms in memory only, never to disk;
  - foundation-model embeddings are reduced to a distance RDM per model in
    memory and discarded - raw embedding vectors are never written;
  - the ONLY persisted output is data/AIS/physics_vessel_transfer_results.csv,
    aggregate scalars only (RSA values, a k-fold probe accuracy, clip/class
    counts). No per-clip rows, no labels, no MMSI, no paths.

What it does
------------
1. Build the 19 frozen foundation-model RDMs on the vessel clips + their
   rank-averaged consensus (per-model RSA is reported too, so the
   "well-spread subset vs all 19" caveat is visible).
2. Run three things trained ONLY on the public probe set, never on vessel:
     - the RDM-distilled physics model  (results/physics_distill_ckpt_phys.pt)
     - the RDM-distilled phys+mel model  (..._phys_mel.pt)
     - the raw PC1 frontend, mean-pooled (no learned params at all)
     - the Wigner-Ville second-moment tensor per clip (Cohen global moments)
3. RSA of each against the vessel consensus + each vessel model, plus a
   group-aware k-fold linear-probe accuracy on the distilled embeddings.

Usage:
    cd data/AIS
    python physics_vessel_transfer.py                # all 19 models + all probes
    python physics_vessel_transfer.py --models clap ast whisper   # subset (debug)
    python physics_vessel_transfer.py --no-probe     # skip the labelled probe
"""
from __future__ import annotations
import argparse, csv, os, sys
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
for p in (REPO, REPO / "scripts", REPO / "scripts" / "physics", HERE):
    sys.path.insert(0, str(p))

import torch
import torch.nn as nn
from scipy.signal import hilbert, resample_poly
from scipy.stats import rankdata, spearmanr
from scipy.spatial.distance import squareform, pdist

from remote_stream import RemoteStream                       # noqa: E402
from train_vessel_classifier_streaming import ClipCache, load_manifest, group_split, NUM_CLASSES  # noqa: E402
from frozen_baseline_streaming import _active_models, embed_all_clips  # noqa: E402
from audio_comp.models import get_model_class                # noqa: E402
import phys_frontend as PF                                   # noqa: E402

RESULT_CSV = HERE / "physics_vessel_transfer_results.csv"
CKPT_DIR = REPO / "results"
SR = PF.SR
EPS = 1e-12


# --------------------------------------------------------------------------- #
# distilled model (must match scripts/physics/step9_rdm_distill.py :: Embed)
# --------------------------------------------------------------------------- #
class Embed(nn.Module):
    def __init__(self, in_dim, emb=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_dim, 64, 5, padding=2), nn.ReLU(),
            nn.Conv1d(64, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv1d(64, emb, 5, stride=2, padding=2), nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x.transpose(1, 2)).mean(-1)


def load_distilled(name):
    p = CKPT_DIR / f"physics_distill_ckpt_{name}.pt"
    if not p.exists():
        print(f"  [skip] {p.name} not found - run scripts/physics/step9_rdm_distill.py first")
        return None
    ck = torch.load(p, map_location="cpu")
    m = Embed(ck["in_dim"], ck["emb"])
    m.load_state_dict(ck["state_dict"])
    m.eval()
    zn = {k: (np.array(v[0], np.float32), np.array(v[1], np.float32))
          for k, v in ck["znorm"].items()}
    return m, zn, ck["frontend"]


# --------------------------------------------------------------------------- #
# WVD second-moment tensor (Cohen global moments) - copy of step10
# --------------------------------------------------------------------------- #
def wvd_tensor(y, sr=SR):
    z = hilbert(y); p = np.abs(z) ** 2; E = p.sum() + EPS
    t = np.arange(len(z)) / sr
    mt = (t * p).sum() / E
    st2 = ((t - mt) ** 2 * p).sum() / E
    ff = np.fft.fftfreq(len(z), 1 / sr); pos = ff >= 0
    pf = (np.abs(np.fft.fft(z)) ** 2)[pos]; f = ff[pos]; Ef = pf.sum() + EPS
    mf = (f * pf).sum() / Ef
    sf2 = ((f - mf) ** 2 * pf).sum() / Ef
    ph = np.unwrap(np.angle(z))
    IF = np.gradient(ph) * sr / (2 * np.pi)
    mif = (IF * p).sum() / E
    if_spread = np.sqrt(max(((IF - mif) ** 2 * p).sum() / E, 0.0))
    cov = (t * IF * p).sum() / E - mt * mif
    T = np.array([[st2, cov / sr], [cov / sr, sf2 / sr ** 2]])
    w = np.linalg.eigvalsh(T)
    aniso = (w[-1] - w[0]) / (w[-1] + w[0] + EPS)
    theta = 0.5 * np.arctan2(2 * T[0, 1], T[0, 0] - T[1, 1])
    chirp = cov / (st2 + EPS)
    return np.array([np.log(st2 + EPS), np.log(sf2 + EPS), cov, chirp, theta,
                     aniso, mif, if_spread], np.float64)


def pad_fix(a, T):
    return a[:T] if len(a) >= T else np.vstack([a, np.zeros((T - len(a), a.shape[1]), a.dtype)])


def frontend_from_wav(y, fe):
    """(T_fix, 65) group-delay+skew+mel-Laplacian ; (T_fix, N_MELS) log-mel."""
    import librosa
    if np.any(y):
        y = librosa.util.normalize(y)
    g16, gsk = PF.group_delay_frames(y)
    lap = PF.spectral_laplacian_frames(y)
    lm = librosa.power_to_db(librosa.feature.melspectrogram(
        y=y, sr=SR, n_fft=fe["N_FFT"], hop_length=fe["HOP"], n_mels=fe["N_MELS"])).T
    Tn = min(len(g16), len(lap), len(lm))
    ph = np.concatenate([g16[:Tn], gsk[:Tn, None], lap[:Tn]], 1).astype(np.float32)
    return pad_fix(np.nan_to_num(ph), fe["T_FIX"]), pad_fix(np.nan_to_num(lm[:Tn].astype(np.float32)), fe["T_FIX"])


def ut(m):
    return m[np.triu_indices(m.shape[0], 1)]


def rdm_corr(X):
    return squareform(pdist(X, metric="correlation"))


# --------------------------------------------------------------------------- #
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=None)
    ap.add_argument("--no-probe", action="store_true")
    ap.add_argument("--kfold", type=int, default=5)
    args = ap.parse_args()

    rows = load_manifest()
    n = len(rows)
    labels = np.array([r["label"] for r in rows])
    groups = np.array([r["vessel_group"] for r in rows])
    print(f"[vessel-transfer] {n} clips, {NUM_CLASSES} classes, "
          f"{len(set(groups))} vessel groups")
    models = args.models or _active_models()

    iu = np.triu_indices(n, 1)

    # ---- 1. foundation-model RDMs + consensus -----------------------------
    with RemoteStream() as remote:
        cache = ClipCache(remote)
        cache.prefetch_all(rows)

        model_rdm_rank = {}
        for mi, mname in enumerate(models):
            adapter = get_model_class(mname)(device="cuda")
            adapter.load()
            emb = embed_all_clips(adapter, mname, rows, cache, list(range(n)))
            model_rdm_rank[mname] = rankdata(ut(rdm_corr(emb)))
            del adapter, emb
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"  [{mi+1}/{len(models)}] {mname} RDM done", flush=True)

        # ---- 2. physics representations on the SAME cached waveforms ------
        print("[vessel-transfer] computing physics frontends / WVD tensors ...")
        fe_any = None
        dists = {name: load_distilled(name) for name in ("phys", "phys_mel")}
        for v in dists.values():
            if v is not None:
                fe_any = v[2]
        if fe_any is None:
            fe_any = {"SR": SR, "N_FFT": PF.N_FFT, "HOP": PF.HOP,
                      "N_MELS": PF.N_MELS, "N_GD_BANDS": PF.N_GD_BANDS,
                      "T_FIX": 400, "MAX_DUR": PF.MAX_DUR}

        ph_frames, mel_frames, wvd_feats, pooled_front = [], [], [], []
        for i, r in enumerate(rows):
            wav, sr = cache.get(r, i)
            y = resample_poly(wav, SR, sr).astype(np.float32) if sr != SR else wav.astype(np.float32)
            y = y[: int(SR * fe_any["MAX_DUR"])]
            pf, mf = frontend_from_wav(y, fe_any)
            ph_frames.append(pf); mel_frames.append(mf)
            pooled_front.append(np.nan_to_num(pf).mean(0))
            wvd_feats.append(wvd_tensor(y))
            if (i + 1) % 200 == 0:
                print(f"  {i+1}/{n}", flush=True)

    ph_frames = np.stack(ph_frames); mel_frames = np.stack(mel_frames)
    pooled_front = np.stack(pooled_front); wvd_feats = np.vstack(wvd_feats)

    # ---- 3. reference RDMs ----------------------------------------------
    consensus = np.mean(list(model_rdm_rank.values()), axis=0)
    # "well-spread subset": the collapsed models (wav2vec2, mms, ...) contribute
    # near-random rank structure to the mean, so we also report RSA against the
    # consensus of just the 6 models most consistent with the full consensus.
    rsa_to_cons = {m: spearmanr(model_rdm_rank[m], consensus).statistic for m in model_rdm_rank}
    top6 = sorted(rsa_to_cons, key=rsa_to_cons.get, reverse=True)[:6]
    consensus_top6 = np.mean([model_rdm_rank[m] for m in top6], axis=0)
    mean_inter_model = np.mean([spearmanr(model_rdm_rank[a], model_rdm_rank[b]).statistic
                                for i, a in enumerate(models) for b in models[i+1:]])

    wvd_z = (wvd_feats - wvd_feats.mean(0)) / (wvd_feats.std(0) + EPS)
    reps = {
        "raw_PC1_frontend_pooled": rankdata(ut(rdm_corr(pooled_front))),
        "WVD_tensor": rankdata(ut(rdm_corr(wvd_z))),
    }
    for name, d in dists.items():
        if d is None:
            continue
        model, zn, fe = d
        if name == "phys":
            X = (ph_frames - zn["phys"][0]) / zn["phys"][1]
        else:  # phys_mel
            X = np.concatenate([(ph_frames - zn["phys"][0]) / zn["phys"][1],
                                (mel_frames - zn["mel"][0]) / zn["mel"][1]], axis=2)
        with torch.no_grad():
            emb = np.vstack([model(torch.tensor(X[b:b+256], dtype=torch.float32)).numpy()
                             for b in range(0, n, 256)])
        reps[f"distilled_{name}"] = rankdata(ut(rdm_corr(emb)))
        reps[f"__emb_{name}"] = emb  # kept locally for the probe, not persisted

    # ---- 4. RSA + probe ------------------------------------------------
    out = []
    for rname, rv in reps.items():
        if rname.startswith("__emb_"):
            continue
        row = dict(representation=rname,
                   rsa_vs_consensus_19=round(spearmanr(rv, consensus).statistic, 3),
                   rsa_vs_consensus_top6=round(spearmanr(rv, consensus_top6).statistic, 3))
        for m in models:
            row[f"rsa_{m}"] = round(spearmanr(rv, model_rdm_rank[m]).statistic, 3)
        out.append(row)

    probe_rows = []
    if not args.no_probe:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import GroupKFold
        gkf = GroupKFold(n_splits=args.kfold)
        for key in [k for k in reps if k.startswith("__emb_")] + ["__pooled__"]:
            E = pooled_front if key == "__pooled__" else reps[key]
            tag = "raw_PC1_frontend_pooled" if key == "__pooled__" else key[6:]
            accs = []
            for tr, te in gkf.split(E, labels, groups):
                sc = StandardScaler().fit(E[tr])
                clf = LogisticRegression(max_iter=2000).fit(sc.transform(E[tr]), labels[tr])
                accs.append(clf.score(sc.transform(E[te]), labels[te]))
            probe_rows.append(dict(representation=tag, kfold=args.kfold,
                                   probe_acc_mean=round(float(np.mean(accs)), 3),
                                   probe_acc_std=round(float(np.std(accs)), 3)))
            print(f"  probe {tag:24s} acc = {np.mean(accs):.3f} +/- {np.std(accs):.3f}")

    # ---- 5. write aggregate-only CSV ---------------------------------
    with open(RESULT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([f"# vessel OOD physics-transfer, n_clips={n}, n_classes={NUM_CLASSES}",
                    f"mean_inter_model_RSA={mean_inter_model:.3f}",
                    f"consensus_top6_models={';'.join(top6)}"])
        keys = list(out[0].keys())
        w.writerow(keys)
        for r in out:
            w.writerow([r[k] for k in keys])
        w.writerow([])
        w.writerow(["representation", "kfold", "probe_acc_mean", "probe_acc_std"])
        for r in probe_rows:
            w.writerow([r["representation"], r["kfold"], r["probe_acc_mean"], r["probe_acc_std"]])

    print(f"\n[vessel-transfer] mean inter-foundation-model RSA on vessel = {mean_inter_model:.3f}")
    print("[vessel-transfer] RSA of each physics representation vs the vessel consensus:")
    for r in out:
        print(f"  {r['representation']:26s} vs-19={r['rsa_vs_consensus_19']:+.3f}  "
              f"vs-top6={r['rsa_vs_consensus_top6']:+.3f}")
    print(f"\n[vessel-transfer] wrote {RESULT_CSV} (aggregate scalars only)")


if __name__ == "__main__":
    main()
