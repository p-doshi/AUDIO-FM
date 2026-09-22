import os
os.environ["OMP_NUM_THREADS"]="1"
os.environ["OPENBLAS_NUM_THREADS"]="1"
os.environ["MKL_NUM_THREADS"]="1"
import sys, gc, numpy as np
from pathlib import Path

def fast_rank(x):
    order = np.argsort(x, kind='mergesort')
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(x) + 1, dtype=np.float64)
    return ranks

CKPT_DIRS_DEFAULT = [Path('/scratch/user/audio_comp/rsa_finetuned_checkpoints'),
                     Path('/scratch/user/audio_comp/.rsa_finetuned_checkpoints_wavlmfix')]

SP = Path('/scratch/user/audio_comp/appendix_work')
RANKCACHE = SP / 'cross18_rank_cache'
RANKCACHE.mkdir(exist_ok=True)

def find_ckpt(domain, cond, model, seed):
    found = None
    for cd in CKPT_DIRS_DEFAULT:
        f = cd / f'{domain}_{cond}_{model}_seed{seed}.npz'
        if f.exists():
            found = f
    return found

if __name__ == '__main__':
    domain, cond, model, seed = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
    cache_path = RANKCACHE / f'{domain}_{cond}_{model}_seed{seed}.npy'
    if cache_path.exists():
        print('already cached')
        sys.exit(0)
    f = find_ckpt(domain, cond, model, int(seed))
    if f is None:
        print('NO CHECKPOINT FOUND')
        sys.exit(1)
    d = np.load(f, allow_pickle=True)
    rdm = d['rdm']
    del d
    iu = np.triu_indices_from(rdm, k=1)
    vec = rdm[iu].astype(np.float64)
    del rdm
    ranked = fast_rank(vec)
    del vec
    np.save(cache_path, ranked)
    print('done')
