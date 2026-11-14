"""Pure model-vs-model RSA (no brain data involved), one 5x5 heatmap per
25%-wide depth quartile (0-25%, 25-50%, 50-75%, 75-100%), using the full
165-stimulus set (not split by category -- that's
`plot_rsa_intercorrelation_by_category.py`'s job; this is the simpler,
overall question of "how much do these 5 models agree with each other at
comparable depths").

Depth binning: models have very different layer counts (wav2vec2/hubert 25,
wavlm/ast 13, whisper 7), so "25% depth" doesn't land on the same layer
index per model. For each model and each quartile, pick the single layer
whose depth fraction is closest to that quartile's midpoint (12.5%, 37.5%,
62.5%, 87.5%) -- picking a representative layer rather than averaging RDMs
across a bucket's layers, since averaging correlation matrices from
structurally different layers would blur rather than represent them.
Whisper's coarse 7-layer resolution means its actual depth per quartile can
land a bit off-center (e.g. ~17% instead of 12.5%) -- shown exactly via
each axis tick's real depth%, not silently rounded to the bucket label.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

BRAIN_RSA_DIR = Path(__file__).resolve().parent
REPO_DIR = BRAIN_RSA_DIR / "auditory_brain_dnn"
DATADIR = REPO_DIR / "data"
ACTV_DIR = BRAIN_RSA_DIR / "activations_per_layer"
PLOTS_DIR = BRAIN_RSA_DIR / "plots"

MODEL_ORDER = ["wav2vec2", "hubert", "wavlm", "whisper", "ast"]
QUARTILES = [(0, 25), (25, 50), (50, 75), (75, 100)]


def correlate_two_matrices_rsa(matrix_1, matrix_2):
    upper_tri = np.triu_indices_from(matrix_1, k=1)
    r, _ = stats.spearmanr(matrix_1[upper_tri], matrix_2[upper_tri])
    return r


def load_stim_order() -> list[str]:
    sound_meta = np.load(DATADIR / "neural" / "NH2015" / "neural_stim_meta.npy")
    return [row[0][:-4].decode("utf-8") for row in sound_meta]


def main() -> None:
    stim_order = load_stim_order()

    # For each model: n_layers, and a lookup from layer_idx -> (165, D) embeddings reordered to stim_order.
    model_data = {}
    for model_name in MODEL_ORDER:
        npz = np.load(ACTV_DIR / f"{model_name}.npz")
        stim_ids = list(npz["stim_ids"])
        idx = {sid: i for i, sid in enumerate(stim_ids)}
        order = [idx[s] for s in stim_order]
        n_layers = int(npz["n_layers"])
        model_data[model_name] = {
            "n_layers": n_layers,
            "layers": {i: npz[f"layer_{i}"][order] for i in range(n_layers)},
        }

    # For each quartile, pick each model's nearest-to-midpoint layer, compute its RDM.
    quartile_matrices = {}
    quartile_depths = {}
    for lo, hi in QUARTILES:
        mid = (lo + hi) / 2
        rdms = {}
        depths = {}
        for model_name in MODEL_ORDER:
            n_layers = model_data[model_name]["n_layers"]
            fracs = [100 * i / (n_layers - 1) if n_layers > 1 else 0.0 for i in range(n_layers)]
            best_layer = int(np.argmin([abs(f - mid) for f in fracs]))
            depths[model_name] = fracs[best_layer]
            embeddings = model_data[model_name]["layers"][best_layer]
            rdms[model_name] = np.corrcoef(embeddings)
        quartile_depths[(lo, hi)] = depths

        n = len(MODEL_ORDER)
        matrix = np.full((n, n), np.nan)
        for i, mi in enumerate(MODEL_ORDER):
            for j, mj in enumerate(MODEL_ORDER):
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j:
                    val = correlate_two_matrices_rsa(rdms[mi], rdms[mj])
                    matrix[i, j] = val
                    matrix[j, i] = val
        quartile_matrices[(lo, hi)] = matrix

    # Shared color scale across all 4 panels (off-diagonal only).
    off_diag_vals = []
    for m in quartile_matrices.values():
        n = m.shape[0]
        off_diag_vals.extend(m[~np.eye(n, dtype=bool)].tolist())
    vmin, vmax = min(off_diag_vals), max(off_diag_vals)

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    axes_flat = axes.flatten()
    im = None

    for panel_idx, (lo, hi) in enumerate(QUARTILES):
        ax = axes_flat[panel_idx]
        matrix = quartile_matrices[(lo, hi)]
        depths = quartile_depths[(lo, hi)]
        labels = [f"{m}\n({depths[m]:.0f}% depth)" for m in MODEL_ORDER]

        im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(len(MODEL_ORDER)))
        ax.set_yticklabels(labels, fontsize=8)

        mid_val = (vmin + vmax) / 2
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                val = matrix[r, c]
                color = "#555555" if r == c else ("white" if val < mid_val else "black")
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=9)

        ax.set_title(f"{lo}-{hi}% depth", fontsize=12)

    fig.tight_layout(rect=[0, 0, 0.92, 0.93])
    fig.suptitle(
        "Model-vs-model RSA at matched depth quartiles (full 165-sound set, no brain data)",
        fontsize=13, y=0.98,
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.03, pad=0.03)
    cbar.set_label(f"RSA (spearman), off-diagonal range {vmin:.2f}-{vmax:.2f}", fontsize=9)

    out_path = PLOTS_DIR / "model_intercorrelation_by_depth_quartile.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
