import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
os.environ["NUMEXPR_NUM_THREADS"]="1"
import sys, gc, numpy as np, pandas as pd, re
from pathlib import Path

def fast_rank(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks

def corr_of_ranked(xr, yr):
    return float(np.corrcoef(xr, yr)[0, 1])

CKPT_DIRS_DEFAULT = [Path('/scratch/user/audio_comp/rsa_finetuned_checkpoints'),
                     Path('/scratch/user/audio_comp/.rsa_finetuned_checkpoints_wavlmfix')]  # last wins (wavlmfix correct for wavlm)

LORA_MODELS = ['ast','bird_mae','clap','data2vec_audio','hubert','mert','mms','music2vec',
               'sew','unispeech_sat','wav2vec2','wav2vec2_conformer','wavlm','whisper']
TOPK_MODELS = ['audio_jepa','audiomae','musicfm','panns_cnn14']

REPO = Path('/home/user/audio_comp')
R = REPO / 'results'
SP = Path('/scratch/user/audio_comp/appendix_work')
OUT = SP / 'cross_18x18'
OUT.mkdir(exist_ok=True)
RANKCACHE = SP / 'cross18_rank_cache'
RANKCACHE.mkdir(exist_ok=True)

def find_ckpt(domain, cond, model, seed):
    found = None
    for cd in CKPT_DIRS_DEFAULT:
        f = cd / f'{domain}_{cond}_{model}_seed{seed}.npz'
        if f.exists():
            found = f
    return found

def get_ranked_triu(domain, cond, model, seed):
    """Load, rank, cache to disk, return path (not array) -- caller loads on demand."""
    cache_path = RANKCACHE / f'{domain}_{cond}_{model}_seed{seed}.npy'
    if cache_path.exists():
        return cache_path
    f = find_ckpt(domain, cond, model, seed)
    if f is None:
        return None
    d = np.load(f, allow_pickle=True)
    rdm = d['rdm']
    del d
    iu = np.triu_indices_from(rdm, k=1)
    vec = rdm[iu].astype(np.float64)
    del rdm
    ranked = fast_rank(vec)
    del vec
    np.save(cache_path, ranked)
    del ranked
    gc.collect()
    return cache_path

def build_cross_block(domain, cond, seed):
    """cond in {'lora','allora'} -- cross with topk"""
    out_path = OUT / f'{domain}_{cond}_topk_cross_seed{seed}.csv'
    if out_path.exists():
        print(f'{domain}/{cond}xtopk seed{seed}: already done, skip', flush=True)
        return

    lora_paths = {}
    for m in LORA_MODELS:
        p = get_ranked_triu(domain, cond, m, seed)
        if p is not None:
            lora_paths[m] = p
            print(f'    {cond}/{m}: ranked+cached', flush=True)

    topk_paths = {}
    for m in TOPK_MODELS:
        p = get_ranked_triu(domain, 'topk', m, seed)
        if p is not None:
            topk_paths[m] = p
            print(f'    topk/{m}: ranked+cached', flush=True)

    rows = []
    for lm, lp in lora_paths.items():
        lv = np.load(lp)
        for tm, tp in topk_paths.items():
            tv = np.load(tp)
            if lv.shape != tv.shape:
                print(f'  SHAPE MISMATCH {domain} {lm} vs {tm} seed{seed}: {lv.shape} vs {tv.shape}', flush=True)
                del tv
                continue
            rho = corr_of_ranked(lv, tv)
            rows.append(dict(lora_model=lm, topk_model=tm, rho=rho))
            del tv
        del lv
        gc.collect()

    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f'{domain}/{cond}xtopk seed{seed}: done, {len(df)} pairs '
          f'({len(lora_paths)} {cond} models x {len(topk_paths)} topk models)', flush=True)

if __name__ == '__main__':
    domain, cond, seed = sys.argv[1], sys.argv[2], int(sys.argv[3])
    build_cross_block(domain, cond, seed)
