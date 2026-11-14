"""MLS cross-lingual RSA. Descriptive extension of Finding 1 to five non-English speech corpora.
Inputs (frozen embeddings, one npz per language and model): $AUDIO_COMP_SCRATCH/mls_crosslingual_emb/{lang}__{model}.npz
Per language:
  1. RDM per model = correlation distance between clip embeddings (paper Eq. 1); clips aligned on the intersection of clip_ids.
  2. 19x19 RSA = Spearman between RDM upper triangles.
  3. Agreement with English: Spearman between the language's RSA upper triangle and LibriSpeech's (results/finetune_data_geometry_rsa_librispeech.csv),
     model-permutation null (10,000, default_rng(0)).
  4. Finding 1 taxonomy Mantel r (Pearson(RSA, same-group)), same grouping and one-sided model-permutation test as rebuild_F1.py (20,000 perms, rng 0).
  5. Uniformity rank agreement with English: Spearman(uniformity_lang, uniformity_librispeech) over models (mls_crosslingual_summary.csv vs
     finetune_data_geometry_matched.csv).
Output: results/additional/mls_crossling/{lang}_rsa_matrix.csv and summary_{lang}.csv"""
import argparse, glob, os, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.environ.get("AUDIO_COMP_ROOT", os.path.expanduser("~/audio_comp")) )
from audio_comp.pipelines.breadth_hypothesis_check import DOMAIN_PARADIGM_GROUPS, BREADTH_GROUPS  # noqa
E = os.environ.get("AUDIO_COMP_SCRATCH", os.path.expanduser("~/scratch/audio_comp")) + "/mls_crosslingual_emb"; R = os.environ.get("AUDIO_COMP_ROOT", os.path.expanduser("~/audio_comp")) + "/results"
O = f"{R}/additional/mls_crossling"; os.makedirs(O, exist_ok=True)
ap = argparse.ArgumentParser(); ap.add_argument("--language", required=True); a = ap.parse_args(); L = a.language

files = sorted(glob.glob(f"{E}/{L}__*.npz")); models = [os.path.basename(f)[len(L) + 2:-4] for f in files]
Z = {m: np.load(f, allow_pickle=True) for m, f in zip(models, files)}
# The extraction saved embeddings only (no clip_ids), in the same deterministic manifest order for every model; decode
# failures do not depend on the model. Align by row position and refuse to run if row counts differ.
if all("clip_ids" in z.files for z in Z.values()):
    common = sorted(set.intersection(*[set(map(str, z["clip_ids"])) for z in Z.values()]))
    rows_of = {m: {c: i for i, c in enumerate(map(str, Z[m]["clip_ids"]))} for m in models}
else:
    counts = {z["embeddings"].shape[0] for z in Z.values()}; assert len(counts) == 1, f"row counts differ: {counts}"
    common = [str(i) for i in range(counts.pop())]; rows_of = {m: {c: int(c) for c in common} for m in models}
print(f"{L}: {len(models)} models, {len(common)} common clips", flush=True)
tri = None; ranks = {}
for m in models:
    pos = rows_of[m]
    X = Z[m]["embeddings"][[pos[c] for c in common]].astype(np.float64)
    X = X - X.mean(1, keepdims=True); X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    D = 1.0 - X @ X.T
    if tri is None: tri = np.triu_indices(len(common), 1)
    ranks[m] = stats.rankdata(D[tri]).astype(np.float32); del D
    print(f"  RDM {m}", flush=True)
n = len(models); M = np.eye(n)
for i in range(n):
    for j in range(i + 1, n):
        M[i, j] = M[j, i] = np.corrcoef(ranks[models[i]], ranks[models[j]])[0, 1]
pd.DataFrame(M, index=models, columns=models).to_csv(f"{O}/{L}_rsa_matrix.csv")

rng = np.random.default_rng(0); iu = np.triu_indices(n, 1)
en = pd.read_csv(f"{R}/finetune_data_geometry_rsa_librispeech.csv", index_col=0)
sh = [m for m in models if m in en.index and m in en.columns]
idx = [models.index(m) for m in sh]; Ml = M[np.ix_(idx, idx)]; Me = en.loc[sh, sh].values.astype(float)
k = len(sh); ku = np.triu_indices(k, 1)
r_en = stats.spearmanr(Ml[ku], Me[ku])[0]
null = []
for _ in range(10000):
    p = rng.permutation(k); null.append(stats.spearmanr(Ml[np.ix_(p, p)][ku], Me[ku])[0])
p_en = (1 + np.sum(np.array(null) >= r_en)) / 10001
rows = [dict(language=L, test="agreement_with_english_rsa", n_models=k, n_clips=len(common), stat=r_en, p=p_en)]

def r_pairs(Mx, g):
    x = Mx[iu]; s = (g[iu[0]] == g[iu[1]]).astype(float)
    return np.corrcoef(x, s)[0, 1] if s.std() > 0 else np.nan
for tax, groups in [("domain+paradigm", DOMAIN_PARADIGM_GROUPS), ("breadth", BREADTH_GROUPS)]:
    gmap = {m: kk for kk, ms in groups.items() for m in ms}; g = np.array([gmap.get(m, f"_single_{m}") for m in models])
    rr = np.random.default_rng(0); obs = r_pairs(M, g)
    nl = np.array([r_pairs(M, g[rr.permutation(n)]) for _ in range(20000)])
    rows.append(dict(language=L, test=f"mantel_{tax}", n_models=n, n_clips=len(common), stat=obs, p=(1 + np.sum(nl >= obs)) / 20001))

s = pd.read_csv(f"{R}/mls_crosslingual_summary.csv"); s = s[s.language == L].set_index("model")
geo = pd.read_csv(f"{R}/finetune_data_geometry_matched.csv"); geo = geo[geo.dataset == "librispeech"].set_index("model")
mm = [m for m in s.index if m in geo.index]
r_u, p_u = stats.spearmanr(s.loc[mm, "uniformity"], geo.loc[mm, "uniformity"])
rows.append(dict(language=L, test="uniformity_rank_vs_english", n_models=len(mm), n_clips=len(common), stat=r_u, p=p_u))
rows.append(dict(language=L, test="mean_offdiag_rsa", n_models=n, n_clips=len(common), stat=M[iu].mean(), p=np.nan))
out = pd.DataFrame(rows); out.to_csv(f"{O}/summary_{L}.csv", index=False); print(out.to_string(index=False), flush=True)
