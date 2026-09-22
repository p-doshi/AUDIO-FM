import numpy as np, pandas as pd
from pathlib import Path

def spearman_rho(x, y):
    xr = pd.Series(x).rank().values
    yr = pd.Series(y).rank().values
    return float(np.corrcoef(xr, yr)[0, 1])

def partial_rho(x, y, z):
    xr = pd.Series(x).rank().values.astype(float)
    yr = pd.Series(y).rank().values.astype(float)
    zr = pd.Series(z).rank().values.astype(float)
    def resid(a, b):
        A = np.vstack([b, np.ones(len(b))]).T
        coef, *_ = np.linalg.lstsq(A, a, rcond=None)
        return a - A @ coef
    return float(np.corrcoef(resid(xr, zr), resid(yr, zr))[0, 1])

def perm_p_partial(x, y, z, n_iter=20000, seed=0):
    rng = np.random.default_rng(seed)
    obs = partial_rho(x, y, z)
    x = np.array(x)
    count = sum(abs(partial_rho(rng.permutation(x), y, z)) >= abs(obs) for _ in range(n_iter))
    return obs, count / n_iter

REPO = Path('/home/user/audio_comp')
R = REPO / 'results'
SP = Path('/scratch/user/audio_comp/appendix_work')
PM = SP / 'per_model_selfrsa'

results_out = []

def run_domain(dom_label, uni_key, res, gain_df, gain_dataset_condition_col='condition', gain_model_col='model', gain_seed_col='seed', gain_acc_col='accuracy', conditions=('lora', 'allora', 'topk'), frozen_label='frozen'):
    uni = pd.read_csv(R / 'finetune_data_geometry_matched.csv')
    uni = uni[uni.dataset == uni_key].set_index('model').uniformity

    if 'fold' in gain_df.columns:
        gain_df = gain_df[gain_df.fold == 'mean'] if (gain_df.fold == 'mean').any() else gain_df
    fr_all = gain_df[gain_df[gain_dataset_condition_col] == frozen_label].groupby([gain_model_col, gain_seed_col])[gain_acc_col].mean()
    ad_map = {c: gain_df[gain_df[gain_dataset_condition_col] == c].groupby([gain_model_col, gain_seed_col])[gain_acc_col].mean() for c in conditions}

    for cond in conditions:
        sub = res[res.condition == cond].copy()
        if sub.empty:
            print(f'  {cond}: no rows', flush=True)
            continue
        ad_all = ad_map[cond]
        sub['uni'] = sub.model.map(uni)
        gains = []
        for _, r in sub.iterrows():
            key = (r.model, r.seed)
            gains.append(ad_all[key] - fr_all[key] if key in ad_all.index and key in fr_all.index else np.nan)
        sub['gain'] = gains
        sub = sub.dropna(subset=['uni', 'gain', 'self_rsa'])
        n = len(sub)
        if n < 6:
            print(f'  {cond:8s} n={n:3d} too small, skipping', flush=True)
            results_out.append(dict(domain=dom_label, condition=cond, n=n, bivariate=np.nan, partial=np.nan, p_partial=np.nan, note='n<6'))
            continue
        biv = spearman_rho(sub.self_rsa, sub.gain)
        par, pp = perm_p_partial(sub.self_rsa.values, sub.gain.values, sub.uni.values)
        print(f'  {cond:8s} n={n:3d}  bivariate={biv:+.3f}   PARTIAL(ctrl uniformity)={par:+.3f} p={pp:.4f}', flush=True)
        results_out.append(dict(domain=dom_label, condition=cond, n=n, bivariate=biv, partial=par, p_partial=pp, note=''))


# ---------------- City / Bird / Machine (already computed, reload for a unified table) ----------------
DOMAIN_GAIN_CSV = {'urbansound8k': 'finetune_urbansound8k.csv', 'birdclef': 'finetune_birdclef.csv'}
for dom in ['urbansound8k', 'birdclef']:
    print(f'\n{"="*20} {dom} {"="*20}', flush=True)
    files = sorted(PM.glob(f'{dom}__*.csv'))
    dfs = [pd.read_csv(f).assign(model=f.stem.split('__', 1)[1]) for f in files if not pd.read_csv(f).empty]
    res = pd.concat(dfs, ignore_index=True)
    gdf = pd.read_csv(R / DOMAIN_GAIN_CSV[dom])
    run_domain(dom, dom, res, gdf, conditions=('lora', 'allora', 'topk'))

print(f'\n{"="*20} mimii {"="*20}', flush=True)
files = sorted(PM.glob('mimii__*.csv'))
dfs = [pd.read_csv(f).assign(model=f.stem.split('__', 1)[1]) for f in files if not pd.read_csv(f).empty]
res = pd.concat(dfs, ignore_index=True)
fr = pd.read_csv(R / 'stage5_frozen_mimii.csv')
accol = 'accuracy' if 'accuracy' in fr.columns else [c for c in fr.columns if 'acc' in c.lower()][0]
fr = fr.rename(columns={accol: 'accuracy'})
fr['condition'] = 'frozen'
lo = pd.read_csv(R / 'stage5_lora_mimii.csv').rename(columns={accol: 'accuracy'}); lo['condition'] = 'lora'
al = pd.read_csv(R / 'stage5_allora_mimii.csv').rename(columns={accol: 'accuracy'}); al['condition'] = 'allora'
tk = pd.read_csv(R / 'stage5_topk_mimii.csv').rename(columns={accol: 'accuracy'}); tk['condition'] = 'topk'
gdf = pd.concat([fr[['model', 'seed', 'condition', 'accuracy']],
                  lo[['model', 'seed', 'condition', 'accuracy']],
                  al[['model', 'seed', 'condition', 'accuracy']],
                  tk[['model', 'seed', 'condition', 'accuracy']]], ignore_index=True)
run_domain('mimii', 'mimii', res, gdf, conditions=('lora', 'allora', 'topk'))

# ---------------- Speech (librispeech) ----------------
print(f'\n{"="*20} speech (librispeech) {"="*20}', flush=True)
res = pd.read_csv(R / 'finding6a_self_rsa_librispeech.csv')
gdf = pd.read_csv(R / 'finetune_librispeech_speaker.csv')
run_domain('librispeech', 'librispeech', res, gdf, conditions=('lora', 'allora', 'topk'))

# ---------------- Music (fma_genre) ----------------
print(f'\n{"="*20} music (fma_genre) {"="*20}', flush=True)
res_main = pd.read_csv(R / 'finding6a_self_rsa_fma_genre.csv')
topk = pd.read_csv(SP / 'fma_topk_selfrsa_fixed.csv')
topk['condition'] = 'topk'
res = pd.concat([res_main, topk[['model', 'condition', 'seed', 'self_rsa']]], ignore_index=True)
gdf = pd.read_csv(R / 'finetune_fma_genre.csv')
run_domain('fma_genre', 'fma_genre', res, gdf, conditions=('lora', 'allora', 'topk'))

# ---------------- Vessel (already has uniformity+gain joined) ----------------
print(f'\n{"="*20} vessel {"="*20}', flush=True)
v = pd.read_csv(R / 'finding6a_self_rsa_vessel_raw_with_uniformity_gain.csv')
for cond in ['lora', 'allora', 'topk']:
    sub = v[v.condition == cond].dropna(subset=['uniformity', 'gain', 'self_rsa'])
    n = len(sub)
    if n < 6:
        print(f'  {cond:8s} n={n:3d} too small, skipping', flush=True)
        results_out.append(dict(domain='vessel', condition=cond, n=n, bivariate=np.nan, partial=np.nan, p_partial=np.nan, note='n<6'))
        continue
    biv = spearman_rho(sub.self_rsa, sub.gain)
    par, pp = perm_p_partial(sub.self_rsa.values, sub.gain.values, sub.uniformity.values)
    print(f'  {cond:8s} n={n:3d}  bivariate={biv:+.3f}   PARTIAL(ctrl uniformity)={par:+.3f} p={pp:.4f}', flush=True)
    results_out.append(dict(domain='vessel', condition=cond, n=n, bivariate=biv, partial=par, p_partial=pp, note=''))

out_df = pd.DataFrame(results_out)
out_df.to_csv(SP / 'partial_corr_FULL_18cell.csv', index=False)
print('\n\n=== FULL 18-CELL TABLE ===')
print(out_df.to_string(index=False))
