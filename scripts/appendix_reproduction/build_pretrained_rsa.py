import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import gc
import numpy as np, pandas as pd
from pathlib import Path
from itertools import combinations

def fast_rank(x):
    # average-tie rank via double argsort, much faster than pandas .rank() for large arrays
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks

def corr_of_ranked(xr, yr):
    return float(np.corrcoef(xr, yr)[0, 1])

MODELS = ['ast','audio_jepa','audiomae','bird_mae','clap','data2vec_audio','encodecmae',
          'hubert','mert','mms','music2vec','musicfm','panns_cnn14','sew','unispeech_sat',
          'wav2vec2','wav2vec2_conformer','wavlm','whisper']

FROZEN_SCRATCH = Path('/scratch/user/audio_comp/finetune_data_geometry_emb')
FROZEN_HOME = Path('/home/user/audio_comp/results/finetune_data_geometry_emb')
SP = Path('/scratch/user/audio_comp/appendix_work')
OUT = SP / 'pretrained_rsa_per_domain'
OUT.mkdir(exist_ok=True)

def get_path(dom, model):
    p = FROZEN_SCRATCH / f'{dom}__{model}.npz'
    if p.exists():
        return p
    return FROZEN_HOME / f'{dom}__{model}.npz'

def build_rdm(emb):
    n = emb.shape[0]
    norm = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    corr = norm @ norm.T
    rdm = 1 - corr
    return rdm

def domain_matrix(dom):
    out_path = OUT / f'{dom}_pretrained_rsa.csv'
    if out_path.exists():
        print(f'{dom}: already done, skip', flush=True)
        return

    # pass 1: determine common clip_id set without holding all embeddings simultaneously
    clip_id_lists = {}
    for m in MODELS:
        p = get_path(dom, m)
        if not p.exists():
            print(f'  {dom}/{m}: missing frozen embedding', flush=True)
            continue
        d = np.load(p, allow_pickle=True)
        clip_id_lists[m] = [str(c) for c in d['clip_ids']]
        del d

    common = None
    for m, cids in clip_id_lists.items():
        s = set(cids)
        common = s if common is None else (common & s)
    common = sorted(common)
    models_present = sorted(clip_id_lists.keys())
    print(f'  {dom}: {len(common)} common clips across {len(models_present)} models', flush=True)

    # pass 2: process one model at a time -- build RDM, rank upper-tri, save to disk immediately,
    # discard from memory. Resumable: skip any model whose rank vector is already cached on disk.
    rank_dir = OUT / f'{dom}_ranks'
    rank_dir.mkdir(exist_ok=True)
    for m in models_present:
        rank_path = rank_dir / f'{m}.npy'
        if rank_path.exists():
            print(f'    {m}: already ranked (cached), skip', flush=True)
            continue
        p = get_path(dom, m)
        d = np.load(p, allow_pickle=True)
        cids = [str(c) for c in d['clip_ids']]
        idx_map = {c: i for i, c in enumerate(cids)}
        order = [idx_map[c] for c in common]
        emb = d['embeddings'].astype(np.float64)[order]
        del d
        rdm = build_rdm(emb)
        del emb
        iu = np.triu_indices_from(rdm, k=1)
        ranked = fast_rank(rdm[iu])
        del rdm
        np.save(rank_path, ranked)
        del ranked
        gc.collect()
        print(f'    {m}: ranked and saved', flush=True)

    mat = pd.DataFrame(index=models_present, columns=models_present, dtype=float)
    for a, b in combinations(models_present, 2):
        ra = np.load(rank_dir / f'{a}.npy')
        rb = np.load(rank_dir / f'{b}.npy')
        rho = corr_of_ranked(ra, rb)
        del ra, rb
        mat.loc[a, b] = rho
        mat.loc[b, a] = rho
    for m in models_present:
        mat.loc[m, m] = 1.0

    mat.to_csv(out_path)
    print(f'{dom}: done, {len(models_present)}x{len(models_present)} matrix', flush=True)

if __name__ == '__main__':
    import sys
    dom = sys.argv[1]
    domain_matrix(dom)
