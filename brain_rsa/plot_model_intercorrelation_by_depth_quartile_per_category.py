"""Model-vs-model RSA at matched depth quartiles (0-25/25-50/50-75/75-100%),
computed separately per stimulus semantic category -- combines
`plot_model_intercorrelation_by_depth_quartile.py`'s depth-quartile framing
with `plot_rsa_intercorrelation_by_category.py`'s per-category framing (no
brain data here, purely model-vs-model, matching the depth-quartile
script's scope; add brain rows the way the other script does if wanted).

One PNG per category (11 files, `plots/depth_quartile_by_category/<cat>.png`)
rather than one giant 11x4 combined grid -- 44 small 5x5 heatmaps in one
image would make every cell's text unreadable; this keeps each category's
2x2 quartile grid at the same legible size as the earlier all-categories
version.

Layer-quartile assignment (nearest layer to each quartile's midpoint) is
the same fixed choice per model regardless of category -- depth binning is
a property of the model's own architecture, not of which sounds are being
compared, so it stays constant; only the *stimulus subset* (and therefore
the resulting RDMs and RSA values) changes per category.
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
PLOTS_DIR = BRAIN_RSA_DIR / "plots" / "depth_quartile_by_category"

MODEL_ORDER = ["wav2vec2", "hubert", "wavlm", "whisper", "ast"]
QUARTILES = [(0, 25), (25, 50), (50, 75), (75, 100)]

d_sound_category_names = {
    "Music": "Instr. Music", "Song": "Vocal Music", "EngSpeech": "English Speech",
    "ForSpeech": "Foreign Speech", "HumVoc": "NonSpeech Vocal", "AniVoc": "Animal Vocal",
    "HumNonVoc": "Human NonVocal", "AniNonVoc": "Animal NonVocal", "Nature": "Nature",
    "Mechanical": "Mechanical", "EnvSound": "Env. Sounds",
}
sound_category_order = [
    "Music", "Song", "EngSpeech", "ForSpeech", "HumVoc", "AniVoc", "HumNonVoc",
    "AniNonVoc", "Nature", "Mechanical", "EnvSound",
]


def correlate_two_matrices_rsa(matrix_1, matrix_2):
    upper_tri = np.triu_indices_from(matrix_1, k=1)
    r, _ = stats.spearmanr(matrix_1[upper_tri], matrix_2[upper_tri])
    return r


def load_stim_meta() -> tuple[list[str], list[str]]:
    sound_meta = np.load(DATADIR / "neural" / "NH2015" / "neural_stim_meta.npy")
    stim_ids = [row[0][:-4].decode("utf-8") for row in sound_meta]
    categories = [row[4] for row in sound_meta]
    return stim_ids, categories


def main() -> None:
    stim_ids, categories = load_stim_meta()
    categories = np.array(categories)

    # Load all layers for all models once, reordered to stim_ids.
    model_data = {}
    for model_name in MODEL_ORDER:
        npz = np.load(ACTV_DIR / f"{model_name}.npz")
        npz_stim_ids = list(npz["stim_ids"])
        idx = {sid: i for i, sid in enumerate(npz_stim_ids)}
        order = [idx[s] for s in stim_ids]
        n_layers = int(npz["n_layers"])
        model_data[model_name] = {
            "n_layers": n_layers,
            "layers": {i: npz[f"layer_{i}"][order] for i in range(n_layers)},
        }

    # Fixed nearest-to-midpoint layer per model per quartile (category-independent).
    quartile_layer = {}
    quartile_depth = {}
    for lo, hi in QUARTILES:
        mid = (lo + hi) / 2
        quartile_layer[(lo, hi)] = {}
        quartile_depth[(lo, hi)] = {}
        for model_name in MODEL_ORDER:
            n_layers = model_data[model_name]["n_layers"]
            fracs = [100 * i / (n_layers - 1) if n_layers > 1 else 0.0 for i in range(n_layers)]
            best_layer = int(np.argmin([abs(f - mid) for f in fracs]))
            quartile_layer[(lo, hi)][model_name] = best_layer
            quartile_depth[(lo, hi)][model_name] = fracs[best_layer]

    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    written = []

    for cat in sound_category_order:
        cat_mask = categories == cat
        n_stim = int(cat_mask.sum())

        quartile_matrices = {}
        for lo, hi in QUARTILES:
            rdms = {}
            for model_name in MODEL_ORDER:
                layer_idx = quartile_layer[(lo, hi)][model_name]
                embeddings = model_data[model_name]["layers"][layer_idx][cat_mask]
                rdms[model_name] = np.corrcoef(embeddings)

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

        off_diag_vals = []
        for m in quartile_matrices.values():
            n = m.shape[0]
            off_diag_vals.extend(m[~np.eye(n, dtype=bool)].tolist())
        vmin, vmax = min(off_diag_vals), max(off_diag_vals)

        fig, axes = plt.subplots(2, 2, figsize=(9, 9))
        axes_flat = axes.flatten()
        im = None

        for panel_idx, (lo, hi) in enumerate(QUARTILES):
            ax = axes_flat[panel_idx]
            matrix = quartile_matrices[(lo, hi)]
            depths = quartile_depth[(lo, hi)]
            labels = [f"{m}\n({depths[m]:.0f}% depth)" for m in MODEL_ORDER]

            im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            ax.set_xticks(range(len(MODEL_ORDER)))
            ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=7.5)
            ax.set_yticks(range(len(MODEL_ORDER)))
            ax.set_yticklabels(labels, fontsize=7.5)

            mid_val = (vmin + vmax) / 2
            for r in range(matrix.shape[0]):
                for c in range(matrix.shape[1]):
                    val = matrix[r, c]
                    color = "#555555" if r == c else ("white" if val < mid_val else "black")
                    ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=8.5)

            ax.set_title(f"{lo}-{hi}% depth", fontsize=11)

        fig.tight_layout(rect=[0, 0, 0.90, 0.92])
        fig.suptitle(
            f"Model-vs-model RSA at matched depth quartiles\n{d_sound_category_names[cat]} (n={n_stim}), no brain data",
            fontsize=13, y=0.99,
        )
        cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.035, pad=0.03)
        cbar.set_label(f"RSA (spearman), range {vmin:.2f}-{vmax:.2f}", fontsize=8.5)

        out_path = PLOTS_DIR / f"{cat}.png"
        fig.savefig(out_path, dpi=140, bbox_inches="tight")
        plt.close(fig)
        written.append(out_path)
        print(f"[{cat}] wrote {out_path}")

    print(f"\nWrote {len(written)} category files to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
