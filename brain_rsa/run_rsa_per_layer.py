"""Per-layer RSA against NH2015, split by ROI (Primary vs. Lateral/Anterior/
Posterior), to actually test the paper's headline claim: middle layers best
predict primary auditory cortex, deep layers best predict non-primary
cortex. v1 (`run_rsa.py`) only tested the final layer and couldn't test this
at all.

Neural-data-loading logic (get_neural_data_matrix equivalent) follows
auditory_brain_dnn/aud_dnn/analyze/rsa_matrix_calculation_all_models.py's
own approach directly (df_roi_meta.pkl -> voxel_id lookup into
voxel_features_array.npy), not utils.py's get_target() (which is for the
whole-brain n_reps==3 case only, no ROI support) — kept as a second,
deliberately separate loader rather than importing theirs, for the same
h5py/seaborn/matplotlib-avoidance reason documented in run_rsa.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BRAIN_RSA_DIR = Path(__file__).resolve().parent
REPO_DIR = BRAIN_RSA_DIR / "auditory_brain_dnn"
DATADIR = REPO_DIR / "data"
ACTV_DIR = BRAIN_RSA_DIR / "activations_per_layer"

ROI_NAMES = ["Primary", "Lateral", "Anterior", "Posterior"]


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


def load_neural_stim_order() -> list[str]:
    sound_meta = np.load(DATADIR / "neural" / "NH2015" / "neural_stim_meta.npy")
    return [row[0][:-4].decode("utf-8") for row in sound_meta]


def load_nh2015_by_roi(roi_name: str | None) -> dict[str, np.ndarray]:
    """Returns {participant_id: (165, n_voxels) array} for the given ROI
    (or all ROI-labeled voxels if roi_name is None -- 'whole-brain (ROI-labeled only)',
    NOT the same voxel set as run_rsa.py's n_reps==3 whole-brain)."""
    voxel_meta_with_roi = pd.read_pickle(DATADIR / "neural" / "NH2015" / "df_roi_meta.pkl")
    voxel_data_all = np.load(DATADIR / "neural" / "NH2015" / "voxel_features_array.npy")
    voxel_meta_all = np.load(DATADIR / "neural" / "NH2015" / "voxel_features_meta.npy")

    voxel_idx_list = [np.where(voxel_meta_all["voxel_id"] == k)[0][0] for k in voxel_meta_with_roi["voxel_id"]]
    voxel_data = voxel_data_all[:, voxel_idx_list, :]
    voxel_meta = voxel_meta_with_roi

    if roi_name is not None:
        is_in_roi = (voxel_meta["roi_label_general"] == roi_name).to_numpy()
        voxel_meta = voxel_meta[is_in_roi]
        voxel_data = voxel_data[:, is_in_roi, :]

    per_participant = {}
    for subj in np.unique(voxel_meta["subj_idx"]):
        mask = (voxel_meta["subj_idx"] == subj).to_numpy()
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
    stim_order = load_neural_stim_order()
    assert len(stim_order) == 165

    rois_to_test = [None] + ROI_NAMES  # None = all ROI-labeled voxels pooled

    rows = []
    for roi in rois_to_test:
        roi_label = roi if roi is not None else "AllROI"
        brain_by_participant = load_nh2015_by_roi(roi)
        participant_corr_mats = {
            pid: run_correlation_on_feature_matrix(data, corr_type="pearson")
            for pid, data in brain_by_participant.items()
        }
        noise_ceiling_vals = []
        for pid in participant_corr_mats:
            others = np.mean([m for p2, m in participant_corr_mats.items() if p2 != pid], axis=0)
            noise_ceiling_vals.append(correlate_two_matrices_rsa(participant_corr_mats[pid], others, "spearman"))
        noise_ceiling_mean = np.mean(noise_ceiling_vals)
        print(f"[{roi_label}] noise ceiling: {noise_ceiling_mean:.3f}")

        model_files = sorted(ACTV_DIR.glob("*.npz"))
        for f in model_files:
            model_name = f.stem
            npz = np.load(f)
            n_layers = int(npz["n_layers"])

            for layer_idx in range(n_layers):
                embeddings = load_model_layer_embeddings(model_name, layer_idx, stim_order)
                model_corr_mat = run_correlation_on_feature_matrix(embeddings, corr_type="pearson")
                vals = [
                    correlate_two_matrices_rsa(model_corr_mat, brain_mat, "spearman")
                    for brain_mat in participant_corr_mats.values()
                ]
                rows.append({
                    "model": model_name,
                    "layer": layer_idx,
                    "n_layers": n_layers,
                    "layer_frac": layer_idx / (n_layers - 1) if n_layers > 1 else 0.0,
                    "roi": roi_label,
                    "rsa_mean": np.mean(vals),
                    "rsa_sem": stats.sem(vals),
                    "noise_ceiling": noise_ceiling_mean,
                    "rsa_noise_corrected": np.mean(vals) / noise_ceiling_mean,
                })

    df = pd.DataFrame(rows)
    out_path = BRAIN_RSA_DIR / "results_per_layer_nh2015.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote {len(df)} rows to {out_path}")

    print("\nPeak layer per model x ROI (by rsa_mean):")
    peak = df.loc[df.groupby(["model", "roi"])["rsa_mean"].idxmax()]
    print(peak[["model", "roi", "layer", "n_layers", "layer_frac", "rsa_mean", "rsa_noise_corrected"]]
          .sort_values(["roi", "model"]).to_string(index=False))


if __name__ == "__main__":
    main()
