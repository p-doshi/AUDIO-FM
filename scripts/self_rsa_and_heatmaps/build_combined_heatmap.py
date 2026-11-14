import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from pathlib import Path

SP = Path('/scratch/user/audio_comp/appendix_work')
PR = SP / 'pretrained_rsa_per_domain'
R = Path('/home/user/audio_comp/results')

MODEL_DISPLAY = {
    'ast':'AST','audio_jepa':'Audio-JEPA','audiomae':'AudioMAE','bird_mae':'Bird-MAE','clap':'CLAP',
    'data2vec_audio':'data2vec-audio','encodecmae':'EnCodecMAE','hubert':'HuBERT','mert':'MERT',
    'mms':'MMS','music2vec':'music2vec','musicfm':'MusicFM','panns_cnn14':'PANNs','sew':'SEW',
    'unispeech_sat':'UniSpeech-SAT','wav2vec2':'wav2vec2','wav2vec2_conformer':'wav2vec2-Conf',
    'wavlm':'WavLM','whisper':'Whisper',
}
MODEL_ORDER_19 = ['ast','panns_cnn14','whisper','clap','audiomae','bird_mae','encodecmae','hubert',
               'wav2vec2','wav2vec2_conformer','wavlm','unispeech_sat','sew','mms','mert','musicfm',
               'audio_jepa','data2vec_audio','music2vec']
LORA_MODELS = ['ast','bird_mae','clap','data2vec_audio','hubert','mert','mms','music2vec',
               'sew','unispeech_sat','wav2vec2','wav2vec2_conformer','wavlm','whisper']
TOPK_MODELS = ['audio_jepa','audiomae','musicfm','panns_cnn14']

def load_all():
    domains_public = ['fma_genre','urbansound8k','birdclef','librispeech','mimii']
    mats = {d: pd.read_csv(PR / f'{d}_pretrained_rsa.csv', index_col=0) for d in domains_public}
    mats['vessel'] = pd.read_csv(R / 'vessel_rsa_matrix_frozen.csv', index_col=0)
    stacked = np.stack([mats[d].loc[MODEL_ORDER_19, MODEL_ORDER_19].values.astype(float) for d in mats])
    avg_pretrained = pd.DataFrame(np.nanmean(stacked, axis=0), index=MODEL_ORDER_19, columns=MODEL_ORDER_19)

    def load_finetuned_block(cond, models):
        mats_list = []
        for d in domains_public:
            for seed in range(5):
                f = R / f'rsa_finetuned_matrix_{d}_{cond}_seed{seed}.csv'
                if f.exists():
                    mats_list.append(pd.read_csv(f, index_col=0).loc[models, models].values.astype(float))
        for seed in range(5):
            f = R / f'vessel_rsa_matrix_{cond}_seed{seed}.csv'
            if f.exists():
                mats_list.append(pd.read_csv(f, index_col=0).loc[models, models].values.astype(float))
        return pd.DataFrame(np.nanmean(np.stack(mats_list), axis=0), index=models, columns=models)

    lora = load_finetuned_block('lora', LORA_MODELS)
    allora = load_finetuned_block('allora', LORA_MODELS)
    topk = load_finetuned_block('topk', TOPK_MODELS)
    return avg_pretrained, lora, allora, topk

def draw_panel(ax, mat, order, title, fontsize=6):
    mat = mat.loc[order, order]
    vals = mat.values.astype(float)
    off_diag = vals[~np.eye(len(order), dtype=bool)]
    vmin, vmax = np.nanmin(off_diag), np.nanmax(off_diag)
    center = (vmin + vmax) / 2
    norm = TwoSlopeNorm(vmin=vmin, vcenter=center, vmax=vmax)
    cmap = plt.cm.RdYlBu_r
    im = ax.imshow(vals, cmap=cmap, norm=norm)
    labels = [MODEL_DISPLAY[m] for m in order]
    ax.set_xticks(range(len(order))); ax.set_xticklabels(labels, rotation=90, fontsize=fontsize)
    ax.set_yticks(range(len(order))); ax.set_yticklabels(labels, fontsize=fontsize)
    ax.set_title(f'{title}  [{vmin:.3f}, {vmax:.3f}]', fontsize=fontsize+2)
    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=fontsize)
    return im

pretrained, lora, allora, topk = load_all()

fig, axes = plt.subplots(2, 2, figsize=(15, 14))

draw_panel(axes[0,0], pretrained, MODEL_ORDER_19, '(a) Pretrained (frozen), 19x19', fontsize=6)
draw_panel(axes[0,1], lora, LORA_MODELS, '(b) Finetuned -- LoRA, 14x14', fontsize=7)
draw_panel(axes[1,0], allora, LORA_MODELS, '(c) Finetuned -- ALLoRA, 14x14', fontsize=7)
draw_panel(axes[1,1], topk, TOPK_MODELS, '(d) Finetuned -- Top-K, 4x4', fontsize=9)

fig.suptitle('Cross-Model RSA Structure: Pretrained vs. Finetuned (domain-averaged)', fontsize=14, y=0.995)
fig.tight_layout(rect=[0, 0, 1, 0.98])
out = SP / 'heatmap_combined_2x2.png'
fig.savefig(out, dpi=200, bbox_inches='tight')
plt.close(fig)
print(f'saved {out}')
