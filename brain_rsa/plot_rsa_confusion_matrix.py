"""Category x model heatmap of peak RSA (best layer, per category) against
NH2015 fMRI responses -- the summary view of
`results_per_layer_by_category.csv`'s per-layer curves. Category full names
and canonical ordering both taken directly from the paper's own
`aud_dnn/resources.py` (`d_sound_category_names`, `sound_category_order`),
not re-derived, so labels match the source paper's own terminology.

Colored with a sequential single-hue ramp (magnitude metric -- RSA here is
never negative in our data, so no diverging/zero-centered scale is needed,
unlike `compare_models.py`'s RdBu_r for model-vs-model RSA/CKA which can go
negative). vmin/vmax set to the data's actual min/max (not a fixed 0-1
range) so the color range isn't mostly unused headroom -- most cells sit in
a 0.1-0.6 band, and a 0-1 scale would wash out the real spread between them.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BRAIN_RSA_DIR = Path(__file__).resolve().parent

# Copied verbatim from auditory_brain_dnn/aud_dnn/resources.py's
# d_sound_category_names / sound_category_order, rather than imported --
# resources.py does an unconditional `import seaborn` at module load
# (needed only for its own plotting functions, which this script doesn't
# use), same avoidance reason as run_rsa.py/run_rsa_per_layer.py's inlined
# RSA math.
d_sound_category_names = {
    "Music": "Instr. Music",
    "Song": "Vocal Music",
    "EngSpeech": "English Speech",
    "ForSpeech": "Foreign Speech",
    "HumVoc": "NonSpeech Vocal",
    "AniVoc": "Animal Vocal",
    "HumNonVoc": "Human NonVocal",
    "AniNonVoc": "Animal NonVocal",
    "Nature": "Nature",
    "Mechanical": "Mechanical",
    "EnvSound": "Env. Sounds",
}
sound_category_order = [
    "Music", "Song", "EngSpeech", "ForSpeech", "HumVoc", "AniVoc", "HumNonVoc",
    "AniNonVoc", "Nature", "Mechanical", "EnvSound",
]

MODEL_ORDER = ["wav2vec2", "hubert", "wavlm", "whisper", "ast"]


def main() -> None:
    df = pd.read_csv(BRAIN_RSA_DIR / "results_per_layer_by_category.csv")

    # Peak (best-layer) RSA per category x model.
    peak = df.loc[df.groupby(["category", "model"])["rsa_mean"].idxmax()]
    pivot = peak.pivot(index="category", columns="model", values="rsa_mean")

    # Order rows by the paper's own canonical category order, columns fixed.
    row_order = [c for c in sound_category_order if c in pivot.index]
    pivot = pivot.loc[row_order, MODEL_ORDER]
    row_labels = [d_sound_category_names[c] for c in row_order]

    n_stim = df.groupby("category")["n_stimuli"].first()
    row_labels_with_n = [f"{label} (n={n_stim[cat]})" for label, cat in zip(row_labels, row_order)]

    matrix = pivot.to_numpy()
    vmin, vmax = np.nanmin(matrix), np.nanmax(matrix)

    fig, ax = plt.subplots(figsize=(7, 7))
    im = ax.imshow(matrix, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")

    ax.set_xticks(range(len(MODEL_ORDER)))
    ax.set_xticklabels(MODEL_ORDER, rotation=30, ha="right")
    ax.set_yticks(range(len(row_labels_with_n)))
    ax.set_yticklabels(row_labels_with_n)

    # Direct-labeled values, dark/light text chosen for contrast against the cell color.
    norm_mid = (vmin + vmax) / 2
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            val = matrix[i, j]
            text_color = "white" if val < norm_mid else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=text_color, fontsize=9)

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(f"peak RSA (spearman), range {vmin:.2f}-{vmax:.2f}", fontsize=9)

    ax.set_title(
        "Model-vs-brain RSA (NH2015), peak across layers,\nby stimulus category",
        fontsize=12,
    )
    fig.tight_layout()

    out_path = BRAIN_RSA_DIR / "plots" / "rsa_confusion_matrix.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    print(f"Wrote {out_path}")
    print(f"\nColor range (actual data min/max, not 0-1): {vmin:.3f} - {vmax:.3f}")
    print(pivot.round(3).to_string())


if __name__ == "__main__":
    main()
