import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
import sys, numpy as np, pandas as pd, re, csv
from pathlib import Path

def spearman_rho(x, y):
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    return float(np.corrcoef(xr, yr)[0, 1])

REPO = Path('/home/user/audio_comp')
# NOTE: order matters. dict assignment below means the LAST directory in this list
# wins on a (cond, seed) key collision. `.rsa_finetuned_checkpoints_wavlmfix` holds a
# corrected rerun for `wavlm` specifically (the original run had a bug for this model
# only); every other model's checkpoints are byte-identical between the two
# directories, verified directly (see journal.md). wavlmfix must be listed LAST so its
# corrected wavlm checkpoint wins over the stale one in the main directory.
CKPT_DIRS = [Path('/scratch/user/audio_comp/rsa_finetuned_checkpoints'),
             Path('/scratch/user/audio_comp/.rsa_finetuned_checkpoints_wavlmfix')]
FROZEN = Path('/scratch/user/audio_comp/finetune_data_geometry_emb')
OUT_DIR = Path('/scratch/user/audio_comp/appendix_work/per_model_selfrsa')
OUT_DIR.mkdir(exist_ok=True)

def urbansound8k_order():
    m = pd.read_csv(REPO / 'data' / 'urbansound8k_manifest.csv')
    return [Path(p).stem for p in m['file'].tolist()]

def birdclef_order():
    with open(REPO / 'data' / 'birdclef_manifest.csv') as f:
        rows = list(csv.DictReader(f))
    clips_dir = Path('/scratch/user/audio_comp/birdclef_clips')
    stems = []
    for row in rows:
        n_clips = int(row['n_clips'])
        species_dir = clips_dir / row['species']
        for i in range(n_clips):
            p = species_dir / f"{row['row_index']}_{i:03d}.wav"
            if p.exists():
                stems.append(p.stem)
    return stems

def mimii_order():
    m = pd.read_csv(REPO / 'data' / 'mimii_manifest.csv')
    rng = np.random.default_rng(0)
    stems = []
    for (machine, condition), group in sorted(m.groupby(['machine', 'condition'])):
        idx = rng.permutation(len(group))[:300]
        picked = group.iloc[idx]
        stems.extend([Path(p).stem for p in picked['file'].tolist()])
    return stems

ORDER_FN = {'urbansound8k': urbansound8k_order, 'birdclef': birdclef_order, 'mimii': mimii_order}

def main():
    dom, model = sys.argv[1], sys.argv[2]
    out_path = OUT_DIR / f'{dom}__{model}.csv'
    if out_path.exists():
        print(f'{dom}/{model}: already done, skip')
        return

    correct_stems = ORDER_FN[dom]()

    existing = {}
    for cd in CKPT_DIRS:
        for f in cd.glob(f'{dom}_*_{model}_seed*.npz'):
            m = re.match(rf'^{dom}_(?P<cond>lora|allora|topk)_{re.escape(model)}_seed(?P<seed>\d)\.npz$', f.name)
            if m:
                existing[(m.group('cond'), int(m.group('seed')))] = f

    if not existing:
        print(f'{dom}/{model}: no checkpoints found')
        pd.DataFrame(columns=['condition','model','seed','self_rsa']).to_csv(out_path, index=False)
        return

    fp = FROZEN / f'{dom}__{model}.npz'
    if not fp.exists() and dom == 'mimii':
        fp = REPO / 'results' / 'finetune_data_geometry_emb' / f'{dom}__{model}.npz'
    if not fp.exists():
        print(f'{dom}/{model}: no frozen embeddings')
        pd.DataFrame(columns=['condition','model','seed','self_rsa']).to_csv(out_path, index=False)
        return

    d = np.load(fp, allow_pickle=True)
    emb = d['embeddings'].astype(np.float64)
    clip_ids = [Path(str(c)).stem for c in d['clip_ids']]
    id_to_idx = {c: i for i, c in enumerate(clip_ids)}
    reorder_idx = [id_to_idx[s] for s in correct_stems if s in id_to_idx]
    n_matched = len(reorder_idx)
    emb_r = emb[reorder_idx]
    emb_n = emb_r / np.linalg.norm(emb_r, axis=1, keepdims=True)
    emb_c = emb_n - emb_n.mean(axis=1, keepdims=True)
    emb_cn = emb_c / np.linalg.norm(emb_c, axis=1, keepdims=True)
    frdm = 1 - emb_cn @ emb_cn.T

    rows = []
    for (cond, seed), f in sorted(existing.items()):
        try:
            adm = np.load(f, allow_pickle=True)['rdm']
        except Exception as e:
            print(f'  SKIP {f.name}: {e}')
            continue
        if adm.shape != frdm.shape:
            print(f'  SKIP {model} {cond} seed{seed}: shape {adm.shape} vs {frdm.shape} (n_matched={n_matched})')
            continue
        iu = np.triu_indices_from(adm, k=1)
        sr = spearman_rho(frdm[iu], adm[iu])
        rows.append(dict(condition=cond, model=model, seed=seed, self_rsa=sr))

    pd.DataFrame(rows).to_csv(out_path, index=False)
    print(f'{dom}/{model}: done, {len(rows)} rows, n_matched={n_matched}')

if __name__ == '__main__':
    main()
