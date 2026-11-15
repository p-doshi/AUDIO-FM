"""Finding 3 with an MLP probe head instead of the linear probe.

Input: results/robustness/mlp_gain.csv (from mlp_converged_gain.py: frozen, adapted and gain accuracy of a one-hidden-
layer MLP, 512 units, L2 penalty chosen on validation, on the same Gram-recovered features, splits and checkpoints as
the converged linear probe; "white" = whitened features as in the paper, "scaled" = non-whitened).
For each feature variant, a copy of RESULTS is made with the converged-probe accuracies replaced by the MLP ones, and
finding3_gain_regression.py, finding3_transferability.py and robustness_analysis.py (part b) are rerun on it.
Outputs: results/robustness/mlp_head/<variant>_{finding3_regression,finding3_transferability,robust_transferability}.csv
         results/robustness/mlp_head/agreement.csv (linear vs MLP accuracy and gain)
"""
import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from common import RESULTS, load_converged_probe

HERE = Path(__file__).resolve().parent
DST = RESULTS / "robustness" / "mlp_head"
KEY = ["dataset", "model", "condition", "seed"]


def main():
    DST.mkdir(parents=True, exist_ok=True)
    mlp = pd.read_csv(RESULTS / "robustness" / "mlp_gain.csv")
    lin = load_converged_probe()
    x = lin.merge(mlp, on=KEY)
    rows = []
    for tag in ["white", "scaled"]:
        p = x.groupby(["dataset", "model"])[["gain", f"gain_mlp_{tag}"]].mean().reset_index()
        rows.append(dict(features=tag, n_cells=len(x),
                         rho_frozen=spearmanr(x.frozen_acc, x[f"frozen_mlp_{tag}"])[0],
                         rho_gain_seed=spearmanr(x.gain, x[f"gain_mlp_{tag}"])[0],
                         rho_gain_pair=spearmanr(p.gain, p[f"gain_mlp_{tag}"])[0],
                         rho_gain_pair_within_domain=np.mean([spearmanr(g.gain, g[f"gain_mlp_{tag}"])[0] for _, g in p.groupby("dataset")]),
                         frac_positive_linear=(x.gain > 0).mean(), frac_positive_mlp=(x[f"gain_mlp_{tag}"] > 0).mean()))
        with tempfile.TemporaryDirectory() as tmp:
            R = Path(tmp) / "results"
            shutil.copytree(RESULTS, R, ignore=shutil.ignore_patterns("paper_outputs", "mlp_head"))
            (R / "paper_outputs").mkdir()
            for f in glob.glob(str(R / "probes" / "converged" / "converged_probe_gain_*.csv")):
                if "_allora_" in f:
                    os.remove(f)
                    continue
                d = pd.read_csv(f)
                d = d[d.model != "model"].astype({"seed": int})
                d = d.merge(mlp[KEY + [f"{c}_mlp_{tag}" for c in ["frozen", "adapted", "gain"]]], on=KEY, how="inner")
                for c in ["frozen", "adapted", "gain"]:
                    d[f"{c}_white"] = d[f"{c}_mlp_{tag}"]
                d.drop(columns=[c for c in d.columns if "_mlp_" in c]).to_csv(f, index=False)
            env = dict(os.environ, ECHO_RESULTS=str(R), ECHO_OUT=str(R / "paper_outputs"))
            for cmd in [["finding3_gain_regression.py"], ["finding3_transferability.py"], ["robustness_analysis.py", "b"]]:
                subprocess.run([sys.executable, str(HERE / cmd[0])] + cmd[1:], env=env, check=True, cwd=HERE)
            for f in ["finding3_regression.csv", "finding3_transferability.csv", "robust_transferability.csv"]:
                shutil.copy(R / "paper_outputs" / f, DST / f"{tag}_{f}")
    pd.DataFrame(rows).to_csv(DST / "agreement.csv", index=False)
    print(pd.DataFrame(rows).round(3).to_string(index=False))


if __name__ == "__main__":
    main()
