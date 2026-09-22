import numpy as np, pandas as pd
from pathlib import Path

def spearman_rho(x, y):
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    return float(np.corrcoef(xr, yr)[0, 1])

REPO = Path('/home/user/audio_comp')
# CORRECTED precedence: wavlmfix (the actual fix) must win over the stale main dir
CKPT_DIRS = [Path('/scratch/user/audio_comp/rsa_finetuned_checkpoints'),
             Path('/scratch/user/audio_comp/.rsa_finetuned_checkpoints_wavlmfix')]  # last wins
FROZEN_SCRATCH = Path('/scratch/user/audio_comp/finetune_data_geometry_emb')
FROZEN_HOME = REPO / 'results' / 'finetune_data_geometry_emb'
OUT = Path('/scratch/user/audio_comp/appendix_work/per_model_selfrsa')

m = pd.read_csv(REPO / 'data' / 'mimii_manifest.csv')
rng = np.random.default_rng(0)
stems = []
for (machine, condition), group in sorted(m.groupby(['machine', 'condition'])):
    idx = rng.permutation(len(group))[:300]
    picked = group.iloc[idx]
    stems.extend([Path(p).stem for p in picked['file'].tolist()])

model = 'wavlm'
dom = 'mimii'
import re
existing = {}
for cd in CKPT_DIRS:
    for f in cd.glob(f'{dom}_*_{model}_seed*.npz'):
        mm = re.match(rf'^{dom}_(lora|allora|topk)_{model}_seed(\d)\.npz$', f.name)
        if mm:
            existing[(mm.group(1), int(mm.group(2)))] = f
            print('using', f, 'for', mm.group(1), mm.group(2))

fp = FROZEN_SCRATCH / f'{dom}__{model}.npz'
if not fp.exists():
    fp = FROZEN_HOME / f'{dom}__{model}.npz'
d = np.load(fp, allow_pickle=True)
emb = d['embeddings'].astype(np.float64)
clip_ids = [Path(str(c)).stem for c in d['clip_ids']]
id_to_idx = {c: i for i, c in enumerate(clip_ids)}
reorder_idx = [id_to_idx[s] for s in stems if s in id_to_idx]
n_matched = len(reorder_idx)
emb_r = emb[reorder_idx]
emb_n = emb_r / np.linalg.norm(emb_r, axis=1, keepdims=True)
emb_c = emb_n - emb_n.mean(axis=1, keepdims=True)
emb_cn = emb_c / np.linalg.norm(emb_c, axis=1, keepdims=True)
frdm = 1 - emb_cn @ emb_cn.T

rows = []
for (cond, seed), f in sorted(existing.items()):
    adm = np.load(f, allow_pickle=True)['rdm']
    if adm.shape != frdm.shape:
        print('SKIP shape mismatch', cond, seed, adm.shape, frdm.shape)
        continue
    iu = np.triu_indices_from(adm, k=1)
    sr = spearman_rho(frdm[iu], adm[iu])
    rows.append(dict(condition=cond, model=model, seed=seed, self_rsa=sr))

out = pd.DataFrame(rows)
out.to_csv(OUT / f'{dom}__{model}.csv', index=False)
print(out)
print(f'n_matched={n_matched}')
