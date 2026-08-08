"""RSA: Spearman correlation between two RDMs — the project's primary metric
(Kriegeskorte et al. 2008)."""
from __future__ import annotations

import numpy as np
from scipy.stats import spearmanr


def rsa_score(rdm_a: np.ndarray, rdm_b: np.ndarray) -> float:
    """Spearman correlation between the upper-triangular entries of two RDMs.

    Both RDMs must be built from the same clip ordering (i.e. the same
    fixed probe set) for the comparison to be meaningful.
    """
    if rdm_a.shape != rdm_b.shape:
        raise ValueError(f"RDM shape mismatch: {rdm_a.shape} vs {rdm_b.shape}")
    iu = np.triu_indices_from(rdm_a, k=1)
    corr, _ = spearmanr(rdm_a[iu], rdm_b[iu])
    return float(corr)
