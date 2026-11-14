"""Score the frozen multi-grain probe against its pre-specified criterion (written before any run).

Per domain, on normalised accuracy (linear probe primary, kNN reported alongside):
 (1) grain dependence: observed coarse-vs-fine rank correlation across models (mean over
     the 5 resplits of Spearman(coarse_r, fine_r)) must lie below the 2.5th percentile of
     the split-noise distribution: Spearman between two different resplits of the SAME
     level (10 pairs x 2 levels = 20 values).
 (2) organised by paradigm: rank shift (fine rank - coarse rank, rank 1 = best, on
     resplit-averaged accuracy) differs across the 4 taxonomy classes (Kruskal-Wallis
     p < 0.05) AND masked-modeling models gain rank at the fine level relative to
     contrastive + supervised models (lower mean shift). Also reported: speech-masked (7)
     vs rest; to call it paradigm rather than pretraining domain, (2) must hold in a
     domain that is not speech (MIMII).
Support requires (1) and (2). Anything else is null.
Output: results/additional/multigrain_score.csv
"""
import os
import glob, itertools, sys
import numpy as np, pandas as pd
from scipy import stats
sys.path.insert(0, os.environ.get("AUDIO_COMP_ROOT", os.path.expanduser("~/audio_comp")) )
from audio_comp.pipelines.breadth_hypothesis_check import DOMAIN_PARADIGM_GROUPS  # noqa

R = os.environ.get("AUDIO_COMP_ROOT", os.path.expanduser("~/audio_comp")) + "/results"
O = f"{R}/additional"
d = pd.concat([pd.read_csv(f) for f in glob.glob(f"{O}/multigrain_parts/*.csv") + glob.glob(f"{O}/multigrain_ext_parts/*.csv")])
tax = pd.read_csv(f"{R}/model_arch_taxonomy.csv").set_index("model").type
SPEECH_MASKED = set(DOMAIN_PARADIGM_GROUPS["masked_modeling_speech"])
out = []
for dom, g in d.groupby("domain"):
    models = sorted(g.model.unique()); n_models = len(models)
    for metric in ["lin_norm", "knn_norm"]:
        P = g.pivot_table(index=["model"], columns=["level", "resplit"], values=metric)
        rs = sorted(g.resplit.unique())
        obs = np.mean([stats.spearmanr(P[("coarse", r)], P[("fine", r)])[0] for r in rs])
        noise = [stats.spearmanr(P[(lv, a)], P[(lv, b)])[0] for lv in ["coarse", "fine"] for a, b in itertools.combinations(rs, 2)]
        floor = np.percentile(noise, 2.5)
        mc = P["coarse"].mean(1); mf = P["fine"].mean(1)
        shift = mf.rank(ascending=False) - mc.rank(ascending=False)          # negative = better rank at fine
        cls = shift.index.map(tax)
        groups = [shift[cls == c].values for c in sorted(set(cls))]
        kw = stats.kruskal(*groups) if all(len(x) > 0 for x in groups) and len(groups) > 1 else (np.nan, np.nan)
        masked = shift[cls == "Masked modeling and reconstruction"].mean()
        contr_sup = shift[cls.isin(["Cross-modal contrastive", "Supervised/discriminative"])].mean()
        sm = shift[shift.index.isin(SPEECH_MASKED)]; rest = shift[~shift.index.isin(SPEECH_MASKED)]
        mw = stats.mannwhitneyu(sm, rest) if len(sm) and len(rest) else (np.nan, np.nan)
        c1 = bool(obs < floor); c2 = bool(kw[1] < 0.05 and masked < contr_sup)
        row = dict(domain=dom, metric=metric, n_models=n_models, rho_coarse_fine=obs, noise_floor_2p5=floor,
                   noise_median=np.median(noise), crit1_grain=c1, kw_H=kw[0], kw_p=kw[1], mean_shift_masked=masked,
                   mean_shift_contrastive_supervised=contr_sup, crit2_paradigm=c2, speech_masked_shift=sm.mean(),
                   rest_shift=rest.mean(), mw_p_speechmasked_vs_rest=mw[1], supported=c1 and c2,
                   max_abs_shift=float(shift.abs().max()))
        out.append(row)
        print(f"\n=== {dom} / {metric}: {n_models} models")
        print(pd.DataFrame({"coarse": mc.round(3), "fine": mf.round(3), "shift": shift, "class": cls}).sort_values("shift").to_string())
        print({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()})
pd.DataFrame(out).to_csv(f"{O}/multigrain_ext_score.csv", index=False)
