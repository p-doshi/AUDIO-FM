import pandas as pd
import numpy as np
from pathlib import Path

R = Path('/home/user/audio_comp/results')
SP = Path('/scratch/user/audio_comp/appendix_work')

ALL_MODELS = ['ast','audio_jepa','audiomae','bird_mae','clap','data2vec_audio','encodecmae',
              'hubert','mert','mms','music2vec','musicfm','panns_cnn14','sew','unispeech_sat',
              'wav2vec2','wav2vec2_conformer','wavlm','whisper']
DOMAINS = ['speech','music','city','bird','machine','vessel']
CONDITIONS = ['frozen','lora','allora','topk']

records = []  # (model, domain, condition, mean, std, n)

def add_from_simple(df, domain, acc_col='accuracy'):
    for (model, cond), g in df.groupby(['model','condition']):
        records.append((model, domain, cond, g[acc_col].mean(), g[acc_col].std(ddof=0), len(g)))

# music, city, bird, speech
add_from_simple(pd.read_csv(R/'finetune_fma_genre.csv'), 'music')
add_from_simple(pd.read_csv(R/'finetune_urbansound8k.csv'), 'city')
add_from_simple(pd.read_csv(R/'finetune_birdclef.csv'), 'bird')
add_from_simple(pd.read_csv(R/'finetune_librispeech_speaker.csv'), 'speech')

# machine (mimii) -- separate files per condition, fold column = held_out_machine
mimii_files = {'frozen':'stage5_frozen_mimii.csv','lora':'stage5_lora_mimii.csv',
               'allora':'stage5_allora_mimii.csv','topk':'stage5_topk_mimii.csv'}
for cond, fname in mimii_files.items():
    d = pd.read_csv(R/fname)
    for model, g in d.groupby('model'):
        records.append((model, 'machine', cond, g['accuracy'].mean(), g['accuracy'].std(ddof=0), len(g)))

# vessel
v = pd.read_csv(R/'vessel_all_experiments.csv')
v['condition'] = v['condition'].replace({'topk_unfreeze':'topk'})
for (model, cond), g in v.groupby(['model','condition']):
    records.append((model, 'vessel', cond, g['test_acc'].mean(), g['test_acc'].std(ddof=0), len(g)))

df = pd.DataFrame(records, columns=['model','domain','condition','mean','std','n'])
df.to_csv(SP / 'big_accuracy_table_raw.csv', index=False)
print(f'Total records: {len(df)}')
print(df.groupby(['domain','condition']).size())

# sanity: coverage check
print('\n--- coverage matrix (n models per domain x condition) ---')
cov = df.groupby(['domain','condition'])['model'].nunique().unstack()
print(cov)
