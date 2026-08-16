"""RSA between our foundation models' final-layer embeddings and human
auditory-cortex fMRI responses (NH2015, Norman-Haignere et al. 2015 via
Tuckute/Feather/McDermott 2023's auditory_brain_dnn repo).

v1 scope: final layer only (whole-model pooled embedding), whole-brain
(all NH2015 voxels with 3 repetitions, no ROI split), no train/test CV
(we're not choosing a best layer yet — one embedding per model). Reuses
the paper's own correlation-matrix + RSA math (run_correlation_on_feature_matrix,
correlate_two_matrices_rsa from their aud_dnn/analyze script) rather than
reimplementing it, to avoid a subtle methodology mismatch.

Per-participant RSA (not just an averaged brain matrix) so we also get a
per-model score distribution and can compare against each participant's own
noise ceiling (leave-one-out reliability of the neural data itself).
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

BRAIN_RSA_DIR = Path(__file__).resolve().parent
REPO_DIR = BRAIN_RSA_DIR / "auditory_brain_dnn"

# Copied verbatim from auditory_brain_dnn/aud_dnn/analyze/rsa_matrix_calculation_all_models.py
# rather than imported, to avoid pulling in that module's unconditional
# `from utils import get_source_features` -> `import h5py` chain (needed
# only for the paper's in-house Kell2018/ResNet50 activation-loading path,
# which we never touch) plus its seaborn/matplotlib plotting-setup imports.


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

DATADIR = REPO_DIR / "data"
ACTV_DIR = BRAIN_RSA_DIR / "activations"


def load_neural_stim_order() -> list[str]:
    sound_meta = np.load(DATADIR / "neural" / "NH2015" / "neural_stim_meta.npy")
    return [row[0][:-4].decode("utf-8") for row in sound_meta]  # strip '.wav'


def load_nh2015_per_participant(stim_order: list[str]) -> dict[str, np.ndarray]:
    """Returns {participant_id: (165, n_voxels) array}, reindexed to stim_order,
    using only voxels with 3 repetitions (n_reps == 3), matching aud_dnn/utils.py's
    get_target()."""
    voxel_data_all = np.load(DATADIR / "neural" / "NH2015" / "voxel_features_array.npy")
    voxel_meta_all = np.load(DATADIR / "neural" / "NH2015" / "voxel_features_meta.npy")
    is_3 = voxel_meta_all["n_reps"] == 3
    voxel_meta, voxel_data = voxel_meta_all[is_3], voxel_data_all[:, is_3, :]

    per_participant = {}
    for subj in np.unique(voxel_meta["subj_idx"]):
        mask = voxel_meta["subj_idx"] == subj
        # voxel_data: (165 sounds, n_voxels, n_reps) -> mean over reps
        per_participant[f"participant_{subj}"] = voxel_data[:, mask, :].mean(axis=2)
    return per_participant


def load_model_embeddings(model_name: str, stim_order: list[str]) -> np.ndarray:
    npz = np.load(ACTV_DIR / f"{model_name}.npz")
    stim_ids = list(npz["stim_ids"])
    embeddings = npz["embeddings"]
    idx = {sid: i for i, sid in enumerate(stim_ids)}
    order = [idx[s] for s in stim_order]
    return embeddings[order]


def main() -> None:
    stim_order = load_neural_stim_order()
    assert len(stim_order) == 165

    brain_by_participant = load_nh2015_per_participant(stim_order)
    participant_corr_mats = {
        pid: run_correlation_on_feature_matrix(data, corr_type="pearson")
        for pid, data in brain_by_participant.items()
    }

    # Noise ceiling: for each participant, correlate their brain corr matrix
    # against the mean of all *other* participants' brain corr matrices.
    noise_ceiling = {}
    for pid in participant_corr_mats:
        others = np.mean(
            [m for other_pid, m in participant_corr_mats.items() if other_pid != pid],
            axis=0,
        )
        noise_ceiling[pid] = correlate_two_matrices_rsa(
            participant_corr_mats[pid], others, distance_measure="spearman"
        )
    print(f"Noise ceiling (leave-one-out participant reliability): "
          f"mean={np.mean(list(noise_ceiling.values())):.3f} "
          f"sem={stats.sem(list(noise_ceiling.values())):.3f}")

    model_files = sorted(ACTV_DIR.glob("*.npz"))
    if not model_files:
        print(f"No activation files found in {ACTV_DIR} yet — extraction jobs may still be running.")
        return

    rows = []
    for f in model_files:
        model_name = f.stem
        embeddings = load_model_embeddings(model_name, stim_order)
        model_corr_mat = run_correlation_on_feature_matrix(embeddings, corr_type="pearson")

        per_participant_rsa = {
            pid: correlate_two_matrices_rsa(model_corr_mat, brain_mat, distance_measure="spearman")
            for pid, brain_mat in participant_corr_mats.items()
        }
        vals = list(per_participant_rsa.values())
        rows.append({
            "model": model_name,
            "rsa_mean": np.mean(vals),
            "rsa_sem": stats.sem(vals),
            "rsa_noise_corrected": np.mean(vals) / np.mean(list(noise_ceiling.values())),
            "embedding_dim": embeddings.shape[1],
        })
        print(f"[{model_name}] RSA vs NH2015 (final layer, whole-brain, 8 participants): "
              f"{np.mean(vals):.3f} +/- {stats.sem(vals):.3f} "
              f"(noise-corrected: {rows[-1]['rsa_noise_corrected']:.3f})")

    df = pd.DataFrame(rows).sort_values("rsa_mean", ascending=False)
    out_path = BRAIN_RSA_DIR / "results_final_layer_nh2015.csv"
    df.to_csv(out_path, index=False)
    print(f"\nWrote results to {out_path}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
