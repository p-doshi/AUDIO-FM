"""Representational dissimilarity matrix (RDM) construction from embeddings."""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist, squareform


def compute_rdm(embeddings: np.ndarray, metric: str = "correlation") -> np.ndarray:
    """Pairwise dissimilarity matrix from an (n_clips, dim) embedding array.

    `metric='correlation'` (1 - Pearson correlation across dims) is the
    standard RSA choice (Kriegeskorte et al. 2008); `'cosine'` is a common
    alternative worth cross-checking against.
    """
    return squareform(pdist(embeddings, metric=metric))
