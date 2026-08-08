"""Build RDMs from each model's embeddings and compare them: RSA (primary),
CKA (secondary/cross-check), TwoNN intrinsic dimension (structural check).

Usage:
    python -m audio_comp.pipelines.compare_models \
        --embeddings-dir $SCRATCH/audio_comp/embeddings \
        --out-dir results

Run after every extract_embeddings.py job has finished. This is the actual
Phase 1 deliverable (per CLAUDE.md): the RSA matrix is what the decision
rule between Phase 1 and Phase 2 is read off of.
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from audio_comp.geometry.cka import linear_cka
from audio_comp.geometry.intrinsic_dim import twonn_intrinsic_dimension
from audio_comp.geometry.rdm import compute_rdm
from audio_comp.geometry.rsa import rsa_score

REPO_ROOT = Path(__file__).resolve().parents[2]


def load_embeddings(embeddings_dir: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    result = {}
    for path in sorted(embeddings_dir.glob("*.npz")):
        data = np.load(path, allow_pickle=True)
        result[path.stem] = (data["clip_ids"], data["embeddings"])
    if not result:
        raise RuntimeError(f"no .npz embedding files found in {embeddings_dir}")
    return result


def align_to_common_order(embeddings: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict[str, np.ndarray]:
    """Reindex every model's embeddings to a shared clip_id order (the first
    model's order), so RDMs are directly comparable row-for-row."""
    model_names = list(embeddings.keys())
    reference_ids = list(embeddings[model_names[0]][0])
    aligned = {}
    for name, (clip_ids, embeds) in embeddings.items():
        id_to_row = {cid: i for i, cid in enumerate(clip_ids)}
        missing = [cid for cid in reference_ids if cid not in id_to_row]
        if missing:
            raise RuntimeError(f"model '{name}' is missing {len(missing)} clip(s) present in the reference set")
        order = [id_to_row[cid] for cid in reference_ids]
        aligned[name] = embeds[order]
    return aligned


def plot_heatmap(matrix: np.ndarray, labels: list[str], title: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(1 + len(labels), 1 + len(labels)))
    im = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=8)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def write_matrix_csv(matrix: np.ndarray, labels: list[str], out_path: Path) -> None:
    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + labels)
        for label, row in zip(labels, matrix):
            writer.writerow([label] + [f"{v:.4f}" for v in row])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--embeddings-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=REPO_ROOT / "results")
    parser.add_argument("--rdm-metric", default="correlation", help="scipy pdist metric, default 'correlation'")
    args = parser.parse_args()

    raw = load_embeddings(args.embeddings_dir)
    aligned = align_to_common_order(raw)
    labels = sorted(aligned.keys())

    rdms = {name: compute_rdm(aligned[name], metric=args.rdm_metric) for name in labels}

    n = len(labels)
    rsa_matrix = np.eye(n)
    cka_matrix = np.eye(n)
    for i in range(n):
        for j in range(i + 1, n):
            r = rsa_score(rdms[labels[i]], rdms[labels[j]])
            c = linear_cka(aligned[labels[i]], aligned[labels[j]])
            rsa_matrix[i, j] = rsa_matrix[j, i] = r
            cka_matrix[i, j] = cka_matrix[j, i] = c

    intrinsic_dims = {name: twonn_intrinsic_dimension(aligned[name]) for name in labels}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_matrix_csv(rsa_matrix, labels, args.out_dir / "rsa_matrix.csv")
    write_matrix_csv(cka_matrix, labels, args.out_dir / "cka_matrix.csv")
    plot_heatmap(rsa_matrix, labels, "RSA (Spearman correlation between RDMs)", args.out_dir / "rsa_heatmap.png")
    plot_heatmap(cka_matrix, labels, "Linear CKA (secondary check only)", args.out_dir / "cka_heatmap.png")

    with open(args.out_dir / "intrinsic_dimension.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["model", "twonn_intrinsic_dimension"])
        for name in labels:
            writer.writerow([name, f"{intrinsic_dims[name]:.3f}"])

    print(f"models compared: {labels}")
    print(f"wrote RSA/CKA matrices, heatmaps, and intrinsic dimensions to {args.out_dir}")


if __name__ == "__main__":
    main()
