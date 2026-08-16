"""Per-layer RSA against a target neural dataset (NH2015 or B2021), split by
the stimulus set's own semantic category labels (Mechanical, EnvSound,
Music, HumNonVoc, HumVoc, Song, EngSpeech, AniVoc, ForSpeech, AniNonVoc,
Nature — from `neural_stim_meta.npy`'s `cat_assignment` field), instead of
pooling all 165 sounds (speech, music, baby crying, animal calls, ... all
very different "tasks") into one RSA number per model/layer. Averaging
across categories this different was flagged as not meaningful — a model
that's great at predicting responses to speech and bad at music would show
a misleading "medium" pooled score.

Whole-brain only (not the ROI-labeled-only subset run_rsa_per_layer.py
uses), since crossing category x ROI x layer x model would fragment
already-small category subsets (as few as 4 sounds in 'Nature') even
further. RSA/correlation-matrix math is the same two functions copied
verbatim from the paper's own script in run_rsa.py/run_rsa_per_layer.py
(see those files' docstrings for why they're inlined rather than imported).

B2021 (Boebinger et al. 2021, 20 participants, 26,792 voxels) support added
2026-08-13: it has 192 stimuli vs. NH2015's 165, confirmed identical order
for the first 165 (checked directly via `stim_info_v4.mat`'s `stim_names`
against NH2015's `neural_stim_meta.npy`, matching `utils.py`'s own
`get_target()` assertion) so the same 165-stimulus category split applies
unchanged. Unlike NH2015, B2021's `voxel_features_meta.npy` is bare voxel
IDs (no `subj_idx`/`n_reps` fields) — participant grouping comes from
`df_roi_meta.pkl` instead, which covers all 26,792 voxels (not just an
ROI-labeled subset) and has no `n_reps` field to filter on (B2021 uses a
fixed 3 repetitions for every voxel, unlike NH2015's mixed 1-3).

Small-category caveat: an RDM/RSA computed on ~4-13 sounds (Nature,
AniNonVoc, ForSpeech, EngSpeech, HumVoc) is noisy — reported as-is, not
hidden, but treat those curves as exploratory, not a firm result.
"""
from __future__ import annotations

import argparse
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

# dataviz skill's validated categorical palette, slots 1-5, fixed order
MODEL_COLORS = {
    "wav2vec2": "#2a78d6",  # blue
    "hubert": "#eb6834",    # orange
    "wavlm": "#1baf7a",     # aqua
    "whisper": "#eda100",   # yellow
    "ast": "#e87ba4",       # magenta
}
MODEL_ORDER = ["wav2vec2", "hubert", "wavlm", "whisper", "ast"]


def run_correlation_on_feature_matrix(feature_matrix, corr_type="pearson"):
    if corr_type == "pearson":
        return np.corrcoef(feature_matrix)
    elif corr_type == "spearman":
        r, _ = stats.spearmanr(feature_matrix)
        return r


def correlate_two_matrices_rsa(matrix_1, matrix_2, distance_measure="pearson"):
    upper_tri = np.triu_indices_from(matrix_1, k=1)
    if distance_measure == "pearson":
        r, _ = stats.pearsonr(matrix_1[upper_tri], matrix_2[upper_tri])
        return r
    elif distance_measure == "spearman":
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


def load_b2021_per_participant(n_stimuli: int) -> dict[str, np.ndarray]:
    """B2021 has 192 stimuli; caller truncates to the first `n_stimuli`
    (165, matching NH2015's order, confirmed identical -- see module docstring)."""
    voxel_data_all = np.load(DATADIR / "neural" / "B2021" / "voxel_features_array.npy")
    voxel_data_all = voxel_data_all[:n_stimuli, :, :]
    voxel_meta_with_roi = pd.read_pickle(DATADIR / "neural" / "B2021" / "df_roi_meta.pkl")

    per_participant = {}
    for subj in np.unique(voxel_meta_with_roi["subj_idx"]):
        mask = (voxel_meta_with_roi["subj_idx"] == subj).to_numpy()
        per_participant[f"participant_{subj}"] = voxel_data_all[:, mask, :].mean(axis=2)
    return per_participant


def load_model_layer_embeddings(model_name: str, layer_idx: int, stim_order: list[str]) -> np.ndarray:
    npz = np.load(ACTV_DIR / f"{model_name}.npz")
    stim_ids = list(npz["stim_ids"])
    embeddings = npz[f"layer_{layer_idx}"]
    idx = {sid: i for i, sid in enumerate(stim_ids)}
    order = [idx[s] for s in stim_order]
    return embeddings[order]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", choices=["NH2015", "B2021"], default="NH2015")
    args = parser.parse_args()

    stim_ids, categories = load_stim_meta()
    categories = np.array(categories)
    unique_cats = pd.Series(categories).value_counts()  # sorted by count, descending

    if args.target == "NH2015":
        brain_by_participant = load_nh2015_per_participant(stim_ids)  # full 165-sound matrices per participant
    else:
        brain_by_participant = load_b2021_per_participant(len(stim_ids))

    rows = []
    for cat in unique_cats.index:
        cat_mask = categories == cat
        cat_stim_ids = [s for s, m in zip(stim_ids, cat_mask) if m]
        n_stim = len(cat_stim_ids)

        # Subset each participant's voxel data to this category's sounds, then
        # build the correlation matrix on the subset (not the full 165 matrix
        # restricted post-hoc -- Pearson correlation across voxels/units is
        # recomputed from the subset directly, same as the paper's own approach).
        participant_corr_mats = {}
        for pid, data in brain_by_participant.items():
            subset = data[cat_mask, :]
            participant_corr_mats[pid] = run_correlation_on_feature_matrix(subset, corr_type="pearson")

        noise_ceiling_vals = []
        for pid in participant_corr_mats:
            others = np.mean([m for p2, m in participant_corr_mats.items() if p2 != pid], axis=0)
            noise_ceiling_vals.append(correlate_two_matrices_rsa(participant_corr_mats[pid], others, "spearman"))
        noise_ceiling_mean = np.mean(noise_ceiling_vals)

        for model_name in MODEL_ORDER:
            npz = np.load(ACTV_DIR / f"{model_name}.npz")
            n_layers = int(npz["n_layers"])

            for layer_idx in range(n_layers):
                embeddings = load_model_layer_embeddings(model_name, layer_idx, stim_ids)
                subset_embeddings = embeddings[cat_mask]
                model_corr_mat = run_correlation_on_feature_matrix(subset_embeddings, corr_type="pearson")

                vals = [
                    correlate_two_matrices_rsa(model_corr_mat, brain_mat, "spearman")
                    for brain_mat in participant_corr_mats.values()
                ]
                rows.append({
                    "category": cat,
                    "n_stimuli": n_stim,
                    "model": model_name,
                    "layer": layer_idx,
                    "n_layers": n_layers,
                    "layer_pct": 100 * layer_idx / (n_layers - 1) if n_layers > 1 else 0.0,
                    "rsa_mean": np.mean(vals),
                    "rsa_sem": stats.sem(vals),
                    "noise_ceiling": noise_ceiling_mean,
                })

    df = pd.DataFrame(rows)
    suffix = "" if args.target == "NH2015" else "_b2021"
    out_path = BRAIN_RSA_DIR / f"results_per_layer_by_category{suffix}.csv"
    df.to_csv(out_path, index=False)
    print(f"Wrote {len(df)} rows to {out_path}")

    # --- Plot: one subplot per category, x=layer depth %, y=RSA, one line per model ---
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    cats_sorted = list(unique_cats.index)  # already sorted by n_stimuli descending
    n_cats = len(cats_sorted)
    ncols = 4
    nrows = int(np.ceil(n_cats / ncols))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.2 * nrows), squeeze=False)
    axes_flat = axes.flatten()

    for i, cat in enumerate(cats_sorted):
        ax = axes_flat[i]
        cat_df = df[df["category"] == cat]
        n_stim = unique_cats[cat]

        for model_name in MODEL_ORDER:
            model_df = cat_df[cat_df["model"] == model_name].sort_values("layer_pct")
            ax.plot(
                model_df["layer_pct"], model_df["rsa_mean"],
                color=MODEL_COLORS[model_name], linewidth=2, marker="o", markersize=4,
                label=model_name,
            )

        noise_ceiling = cat_df["noise_ceiling"].iloc[0]
        ax.axhline(noise_ceiling, color="#8a8a86", linestyle="--", linewidth=1, label="noise ceiling")

        ax.set_title(f"{cat} (n={n_stim})", fontsize=10)
        ax.set_xlabel("layer depth (%)", fontsize=8)
        ax.set_ylabel("RSA (spearman)", fontsize=8)
        ax.tick_params(labelsize=7)
        ax.set_xlim(-2, 102)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.grid(axis="y", color="#e5e5e0", linewidth=0.6)

    for j in range(n_cats, len(axes_flat)):
        axes_flat[j].set_visible(False)

    handles, labels = axes_flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"RSA vs. layer depth, by stimulus semantic category ({args.target}, whole-brain)", fontsize=13, y=1.01)
    fig.tight_layout(rect=[0, 0.03, 1, 1])

    plot_path = PLOTS_DIR / f"rsa_by_layer_per_category{suffix}.png"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"Wrote plot to {plot_path}")


if __name__ == "__main__":
    main()
