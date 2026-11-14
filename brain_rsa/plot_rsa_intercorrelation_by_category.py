"""Per-category (Brain + models) x (Brain + models) RSA heatmap grid --
extends `plot_rsa_confusion_matrix.py` (which only showed model-vs-brain)
to also show model-vs-model agreement, and folds in layer depth (dropped
from that earlier heatmap) via each model's peak-matching layer shown in
its axis tick label.

Per category, each model contributes the single layer that peaked against
the brain for that category (same "peak across layers" choice
`plot_rsa_confusion_matrix.py` uses) -- one representative RDM per model
per category, not a full per-layer comparison. Model-vs-model cells
correlate those two models' RDMs directly (no brain data involved, so no
participant-averaging needed, unlike the brain-vs-model cells which are the
already-computed cross-participant mean from
`results_per_layer_by_category.csv`).

Color scale is shared across all 11 category panels (one global vmin/vmax,
not per-panel), so panels are visually comparable to each other -- a category
where everything correlates highly should look visibly "hotter" than one
where nothing does, which a per-panel-normalized scale would hide.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

BRAIN_RSA_DIR = Path(__file__).resolve().parent
REPO_DIR = BRAIN_RSA_DIR / "auditory_brain_dnn"
DATADIR = REPO_DIR / "data"
ACTV_DIR = BRAIN_RSA_DIR / "activations_per_layer"
PLOTS_DIR = BRAIN_RSA_DIR / "plots"

MODEL_ORDER = ["wav2vec2", "hubert", "wavlm", "whisper", "ast"]
NODE_ORDER = ["Brain"] + MODEL_ORDER

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


def run_correlation_on_feature_matrix(feature_matrix, corr_type="pearson"):
    if corr_type == "pearson":
        return np.corrcoef(feature_matrix)


def correlate_two_matrices_rsa(matrix_1, matrix_2, distance_measure="spearman"):
    upper_tri = np.triu_indices_from(matrix_1, k=1)
    if distance_measure == "spearman":
        r, _ = stats.spearmanr(matrix_1[upper_tri], matrix_2[upper_tri])
        return r


def load_stim_meta() -> tuple[list[str], list[str]]:
    sound_meta = np.load(DATADIR / "neural" / "NH2015" / "neural_stim_meta.npy")
    stim_ids = [row[0][:-4].decode("utf-8") for row in sound_meta]
    categories = [row[4] for row in sound_meta]
    return stim_ids, categories


def load_nh2015_per_participant(stim_order: list[str]) -> dict[str, np.ndarray]:
    voxel_data_all = np.load(DATADIR / "neural" / "NH2015" / "voxel_features_array.npy")
    voxel_meta_all = np.load(DATADIR / "neural" / "NH2015" / "voxel_features_meta.npy")
    is_3 = voxel_meta_all["n_reps"] == 3
    voxel_meta, voxel_data = voxel_meta_all[is_3], voxel_data_all[:, is_3, :]
    per_participant = {}
    for subj in np.unique(voxel_meta["subj_idx"]):
        mask = voxel_meta["subj_idx"] == subj
        per_participant[f"participant_{subj}"] = voxel_data[:, mask, :].mean(axis=2)
    return per_participant


def load_model_layer_embeddings(model_name: str, layer_idx: int, stim_order: list[str]) -> np.ndarray:
    npz = np.load(ACTV_DIR / f"{model_name}.npz")
    stim_ids = list(npz["stim_ids"])
    embeddings = npz[f"layer_{layer_idx}"]
    idx = {sid: i for i, sid in enumerate(stim_ids)}
    order = [idx[s] for s in stim_order]
    return embeddings[order]


def main() -> None:
    stim_ids, categories = load_stim_meta()
    categories = np.array(categories)
    brain_by_participant = load_nh2015_per_participant(stim_ids)

    per_layer_df = pd.read_csv(BRAIN_RSA_DIR / "results_per_layer_by_category.csv")
    # Each model's peak (brain-best) layer + depth%, per category.
    peak_df = per_layer_df.loc[per_layer_df.groupby(["category", "model"])["rsa_mean"].idxmax()]

    all_matrices: dict[str, np.ndarray] = {}
    all_depths: dict[str, dict[str, float]] = {}
    n_stim_by_cat: dict[str, int] = {}

    for cat in sound_category_order:
        cat_mask = categories == cat
        n_stim = int(cat_mask.sum())
        n_stim_by_cat[cat] = n_stim

        # Brain RDM (participant-averaged, matching earlier scripts' methodology)
        participant_corr_mats = {
            pid: run_correlation_on_feature_matrix(data[cat_mask, :], corr_type="pearson")
            for pid, data in brain_by_participant.items()
        }

        # Each model's peak-layer RDM for this category
        model_corr_mats = {}
        depths = {}
        for model_name in MODEL_ORDER:
            row = peak_df[(peak_df["category"] == cat) & (peak_df["model"] == model_name)].iloc[0]
            layer_idx = int(row["layer"])
            depths[model_name] = float(row["layer_pct"])
            embeddings = load_model_layer_embeddings(model_name, layer_idx, stim_ids)[cat_mask]
            model_corr_mats[model_name] = run_correlation_on_feature_matrix(embeddings, corr_type="pearson")
        all_depths[cat] = depths

        n = len(NODE_ORDER)
        matrix = np.full((n, n), np.nan)
        for i, node_i in enumerate(NODE_ORDER):
            for j, node_j in enumerate(NODE_ORDER):
                if i == j:
                    matrix[i, j] = 1.0
                elif i < j:
                    if node_i == "Brain":
                        rdm_j = model_corr_mats[node_j]
                        vals = [correlate_two_matrices_rsa(rdm_j, bm) for bm in participant_corr_mats.values()]
                        val = np.mean(vals)
                    else:
                        val = correlate_two_matrices_rsa(model_corr_mats[node_i], model_corr_mats[node_j])
                    matrix[i, j] = val
                    matrix[j, i] = val
        all_matrices[cat] = matrix

    # Shared color scale across all category panels (off-diagonal cells only)
    off_diag_vals = []
    for m in all_matrices.values():
        n = m.shape[0]
        off_diag_vals.extend(m[~np.eye(n, dtype=bool)].tolist())
    vmin, vmax = np.nanmin(off_diag_vals), np.nanmax(off_diag_vals)

    ncols = 4
    nrows = int(np.ceil(len(sound_category_order) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.6 * ncols, 4.6 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    im = None
    for i, cat in enumerate(sound_category_order):
        ax = axes_flat[i]
        matrix = all_matrices[cat]
        depths = all_depths[cat]
        labels = ["Brain"] + [f"{m}\n({depths[m]:.0f}% depth)" for m in MODEL_ORDER]

        im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(NODE_ORDER)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6.5)
        ax.set_yticks(range(len(NODE_ORDER)))
        ax.set_yticklabels(labels, fontsize=6.5)

        mid = (vmin + vmax) / 2
        for r in range(matrix.shape[0]):
            for c in range(matrix.shape[1]):
                val = matrix[r, c]
                color = "white" if (val < mid and r != c) else "black"
                if r == c:
                    color = "#555555"
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", color=color, fontsize=6.5)

        ax.set_title(f"{d_sound_category_names[cat]} (n={n_stim_by_cat[cat]})", fontsize=10)

    for j in range(len(sound_category_order), len(axes_flat)):
        axes_flat[j].set_visible(False)

    fig.tight_layout(rect=[0, 0, 0.94, 0.95])
    fig.suptitle(
        "Model-vs-model and model-vs-brain RSA per category (NH2015, whole-brain)\n"
        "each model shown at its own peak (brain-best) layer depth for that category",
        fontsize=13, y=0.99,
    )
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.02)
    cbar.set_label(f"RSA (spearman), off-diagonal range {vmin:.2f}-{vmax:.2f}", fontsize=9)

    out_path = PLOTS_DIR / "rsa_intercorrelation_by_category.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
