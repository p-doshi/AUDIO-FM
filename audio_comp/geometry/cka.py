"""Linear CKA (Kornblith et al., ICML 2019) — secondary/cross-check metric
only. Known to be gameable (Davari et al., ICLR 2023) — never report CKA
alone, always pair with RSA (see rsa.py)."""
from __future__ import annotations

import numpy as np


def _center_gram(gram: np.ndarray) -> np.ndarray:
    n = gram.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    return centering @ gram @ centering


def linear_cka(x: np.ndarray, y: np.ndarray) -> float:
    """Linear CKA between two (n_clips, dim) embedding arrays with matched rows."""
    if x.shape[0] != y.shape[0]:
        raise ValueError(f"row count mismatch: {x.shape[0]} vs {y.shape[0]}")
    gram_x = _center_gram(x @ x.T)
    gram_y = _center_gram(y @ y.T)
    numerator = np.sum(gram_x * gram_y)
    denominator = np.sqrt(np.sum(gram_x * gram_x) * np.sum(gram_y * gram_y))
    return float(numerator / denominator) if denominator != 0 else float("nan")
