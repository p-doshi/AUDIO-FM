"""Intrinsic dimension (TwoNN, Facco et al. 2017) per model — an independent
structural check, useful for interpreting *why* two models might disagree
(e.g. one collapsed to a much lower intrinsic dimension than the other)."""
from __future__ import annotations

import numpy as np
import skdim


def twonn_intrinsic_dimension(embeddings: np.ndarray) -> float:
    estimator = skdim.id.TwoNN()
    estimator.fit(embeddings)
    return float(estimator.dimension_)
