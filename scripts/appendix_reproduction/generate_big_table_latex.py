import pandas as pd
from pathlib import Path

SP = Path('/scratch/user/audio_comp/appendix_work')
df = pd.read_csv(SP / 'big_accuracy_table_raw.csv')

MODEL_DISPLAY = {
    'ast':'AST','audio_jepa':'Audio-JEPA','audiomae':'AudioMAE','bird_mae':'Bird-MAE','clap':'CLAP',
    'data2vec_audio':'data2vec-audio','encodecmae':'EnCodecMAE','hubert':'HuBERT','mert':'MERT',
    'mms':'MMS','music2vec':'music2vec','musicfm':'MusicFM','panns_cnn14':'PANNs (CNN14)','sew':'SEW',
    'unispeech_sat':'UniSpeech-SAT','wav2vec2':'wav2vec 2.0','wav2vec2_conformer':'wav2vec 2.0 Conf.',
    'wavlm':'WavLM','whisper':'Whisper',
}
MODEL_ORDER = ['ast','panns_cnn14','whisper','clap','audiomae','bird_mae','encodecmae','hubert',
               'wav2vec2','wav2vec2_conformer','wavlm','unispeech_sat','sew','mms','mert','musicfm',
               'audio_jepa','data2vec_audio','music2vec']

DOMAINS = ['speech','music','city','bird','machine','vessel']
DOMAIN_DISPLAY = {'speech':'Speech','music':'Music','city':'City','bird':'Bird','machine':'Machine','vessel':'Vessel'}
CONDS = ['frozen','lora','allora','topk']
COND_DISPLAY = {'frozen':'Fz','lora':'LoRA','allora':'ALoRA','topk':'TopK'}

lookup = {}
for _, r in df.iterrows():
    lookup[(r['model'], r['domain'], r['condition'])] = (r['mean'], r['std'], r['n'])

def cell(model, domain, cond):
    key = (model, domain, cond)
    if key not in lookup:
        return '--'
    mean, std, n = lookup[key]
    return f'{mean:.2f}$\\pm${std:.2f}'

lines = []
lines.append(r'\begin{sidewaystable}[p]')
lines.append(r'\centering')
lines.append(r'\tiny')
lines.append(r'\setlength{\tabcolsep}{2.2pt}')
ncols = 1 + len(DOMAINS)*len(CONDS)
colspec = 'l' + 'cccc'*len(DOMAINS)
lines.append(r'\resizebox{\textwidth}{!}{%')
lines.append(r'\begin{tabular}{' + colspec + '}')
lines.append(r'\toprule')
# top header row: domain names spanning 4 cols each
header1 = ['Model'] + [f'\\multicolumn{{4}}{{c}}{{{DOMAIN_DISPLAY[d]}}}' for d in DOMAINS]
lines.append(' & '.join(header1) + r' \\')
# cmidrules
cmid = []
start = 2
for d in DOMAINS:
    cmid.append(f'\\cmidrule(lr){{{start}-{start+3}}}')
    start += 4
lines.append(''.join(cmid))
# second header row: condition labels
header2 = [''] + [COND_DISPLAY[c] for d in DOMAINS for c in CONDS]
lines.append(' & '.join(header2) + r' \\')
lines.append(r'\midrule')

for m in MODEL_ORDER:
    row = [MODEL_DISPLAY[m]]
    for d in DOMAINS:
        for c in CONDS:
            row.append(cell(m, d, c))
    lines.append(' & '.join(row) + r' \\')

lines.append(r'\bottomrule')
lines.append(r'\end{tabular}%')
lines.append(r'}')
lines.append(r'\caption{Accuracy (mean$\pm$std across seeds, and across folds for city/bird/machine) for every model, domain, and adaptation condition. Fz = frozen linear probe; LoRA/ALoRA/TopK = adapted accuracy under that condition. \texttt{--} indicates the model is not eligible for that adaptation condition (Section~\ref{sec:modelroster}): 14 models receive LoRA/ALLoRA, 4 receive top-$K$ only, and EnCodecMAE is frozen-only. City and bird accuracies are averaged over folds then seeds; machine (MIMII) is averaged over held-out-machine-type folds then seeds; vessel is averaged over seeds only, under the governing data-sharing agreement.}')
lines.append(r'\label{tab:big_accuracy}')
lines.append(r'\end{sidewaystable}')

out = '\n'.join(lines)
(SP / 'big_accuracy_table.tex').write_text(out)
print(out[:2000])
print('...')
print(f'\nTotal lines: {len(lines)}')
