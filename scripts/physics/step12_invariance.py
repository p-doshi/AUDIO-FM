"""Step 12 - two runnable tests of physical invariance for the PC1 frontend.

(A) Sample efficiency: 8-way category task, training labels capped at
    k in {5,10,20,40,80,160,300} per class, 3 seeds. Does phys / phys+mel
    reach target accuracy with fewer labels than logmel?

(B) Acoustic-condition shift: train a 6-way classifier on the clean/native
    categories {music, speech, bird_sounds, ship_vessel, city_noise,
    machine_sounds}; test (i) on held-out clean clips and (ii) on
    speech_noisy clips relabelled 'speech' and music_noisy clips relabelled
    'music' - i.e. the SAME source content through a different acoustic
    channel (AMI far-field meeting room / SingVERSE added noise). The
    accuracy drop from (i) to (ii) is the (in)variance gap. Smaller drop for
    phys => the representation is more invariant to the propagation channel.

Reuses the Step-8 feature cache. Output: results/physics_invariance.csv
"""
import os, numpy as np, pandas as pd, torch
import torch.nn as nn
from step8_phys_arch import CACHE, build_cache, Head
import common as C

torch.manual_seed(0); np.random.seed(0)
CLEAN6 = ["music", "speech", "bird_sounds", "ship_vessel", "city_noise", "machine_sounds"]
NOISY_MAP = {"speech_noisy": "speech", "music_noisy": "music"}


def load():
    build_cache()
    d = np.load(CACHE, allow_pickle=True)
    return (np.nan_to_num(d["phys"]), np.nan_to_num(d["mel"]),
            d["categories"], d["clip_ids"])


def znorm(X, tr):
    f = X[tr].reshape(-1, X.shape[2])
    mu, sd = f.mean(0), f.std(0) + 1e-6
    return ((X - mu) / sd).astype(np.float32)


def train_eval(Xtr, ytr, test_sets, n_cls, epochs=40, seed=0):
    """test_sets: dict name -> (X, y). Returns dict name -> accuracy."""
    torch.manual_seed(seed)
    m = Head(Xtr.shape[2], n_cls=n_cls)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3, weight_decay=1e-4)
    lf = nn.CrossEntropyLoss()
    dt, yt = torch.tensor(Xtr), torch.tensor(ytr)
    for ep in range(epochs):
        m.train(); perm = torch.randperm(len(dt))
        for b in range(0, len(dt), 64):
            i = perm[b:b + 64]
            opt.zero_grad(); loss = lf(m(dt[i])[0], yt[i])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 5.0); opt.step()
    m.eval()
    out = {}
    with torch.no_grad():
        for name, (Xe, ye) in test_sets.items():
            pred = m(torch.tensor(Xe))[0].argmax(1).numpy()
            out[name] = float((pred == ye).mean())
    return out


def main():
    phys, mel, cats, ids = load()
    conds = {"phys": phys, "logmel": mel, "phys+mel": None}
    rng = np.random.RandomState(0)

    # fixed test split (15% per category, all 8 cats)
    classes8 = sorted(set(cats)); c8 = {c: i for i, c in enumerate(classes8)}
    y8 = np.array([c8[c] for c in cats])
    te, pool = [], []
    for c in range(8):
        idx = np.where(y8 == c)[0]; rng.shuffle(idx)
        k = int(.15 * len(idx)); te += idx[:k].tolist(); pool += idx[k:].tolist()
    te = np.array(te); pool = np.array(pool)

    def feats(name, base_tr):
        if name == "phys+mel":
            return np.concatenate([znorm(phys, base_tr), znorm(mel, base_tr)], 2)
        return znorm(phys if name == "phys" else mel, base_tr)

    rows = []
    # ---- (A) sample efficiency ----
    for k in [5, 10, 20, 40, 80, 160, 300]:
        for seed in range(3):
            r = np.random.RandomState(seed)
            tr = []
            for c in range(8):
                pc = pool[y8[pool] == c]; r.shuffle(pc); tr += pc[:k].tolist()
            tr = np.array(tr)
            for name in conds:
                X = feats(name, tr)
                acc = train_eval(X[tr], y8[tr], {"t": (X[te], y8[te])}, 8, seed=seed)["t"]
                rows.append(dict(test="sample_efficiency", frontend=name,
                                 labels_per_class=k, seed=seed, acc=round(acc, 4)))
        cur = pd.DataFrame(rows)
        piv = cur[cur.labels_per_class == k].groupby("frontend").acc.mean()
        print(f"  k={k:4d}/class  " + "  ".join(f"{n}={piv[n]:.3f}" for n in conds))

    # ---- (B) acoustic-condition shift ----
    print("\n[step12] (B) acoustic-condition shift (6-way, clean->noisy channel)")
    cl = {c: i for i, c in enumerate(CLEAN6)}
    is_clean = np.array([c in cl for c in cats])
    is_noisy = np.array([c in NOISY_MAP for c in cats])
    ycl = np.array([cl[c] if c in cl else -1 for c in cats])
    ynz = np.array([cl[NOISY_MAP[c]] if c in NOISY_MAP else -1 for c in cats])
    for seed in range(3):
        r = np.random.RandomState(seed)
        cl_idx = np.where(is_clean)[0]; r.shuffle(cl_idx)
        n_te = int(.2 * len(cl_idx))
        cte, ctr = cl_idx[:n_te], cl_idx[n_te:]
        nz_idx = np.where(is_noisy)[0]
        for name in conds:
            X = feats(name, ctr)
            res = train_eval(X[ctr], ycl[ctr], {
                "clean_test": (X[cte], ycl[cte]),
                "shifted_channel": (X[nz_idx], ynz[nz_idx]),
            }, 6, seed=seed)
            gap = res["clean_test"] - res["shifted_channel"]
            rows.append(dict(test="condition_shift", frontend=name, seed=seed,
                             acc=round(res["clean_test"], 4),
                             acc_shifted=round(res["shifted_channel"], 4),
                             invariance_gap=round(gap, 4)))
    df = pd.DataFrame(rows)
    df.to_csv(f"{C.OUT}/physics_invariance.csv", index=False)

    b = df[df.test == "condition_shift"].groupby("frontend")[["acc", "acc_shifted", "invariance_gap"]].mean()
    print(b.round(3).to_string())
    print("\n[step12] sample-efficiency curve (mean acc over 3 seeds):")
    a = df[df.test == "sample_efficiency"].pivot_table(index="labels_per_class", columns="frontend", values="acc")
    print(a.round(3).to_string())
    print(f"\n[step12] saved {C.OUT}/physics_invariance.csv")


if __name__ == "__main__":
    main()
