"""Wang & Isola (ICML 2020) alignment/uniformity, adapted to this
project's probe set as the real Stage 6 diagnostic candidate flagged
throughout Stage 1 (replacing the earlier silhouette-score proxy).

Original definitions (both on the unit hypersphere, lower = better for a
contrastively-trained model):
  alignment(f) = E_{(x,y) ~ p_pos}[ ||f(x)-f(y)||_2^2 ]
  uniformity(f) = log E_{x,y iid p_data}[ exp(-t * ||f(x)-f(y)||_2^2) ]

The original paper defines "positive pairs" as augmented views of the
same instance (from contrastive-learning setups where that pairing is
given by construction). This project's probe set has no such augmented-
view structure, so **alignment here is adapted to use same-category
clips as the positive-pair proxy** (5 categories: music, speech,
bird_sounds, ship_vessel, city_noise) — a reasonable stand-in for "should
be close together" but not the literal original definition; state this
plainly wherever these numbers are reported, don't imply they're the
textbook metric unmodified.

Both metrics require L2-normalized (unit hypersphere) embeddings; this
module normalizes internally, callers pass raw embeddings.

Efficiency: alignment is computed via the centroid trick (mean pairwise
cosine similarity of n unit vectors = (||sum||^2 - n) / (n(n-1)),
avoiding O(n^2) work per category) since every category here has exactly
2000 clips. Uniformity is estimated on a fixed-seed random subsample
(default 3000 clips) rather than the full 10,000 -- an unbiased estimate
of the same expectation at a fraction of the O(n^2) pairwise-distance
cost, matching the "correctness over exhaustiveness for a first working
integration" convention already used in xares_eval's encoder wrappers.
"""
from __future__ import annotations

import numpy as np
from scipy.spatial.distance import pdist


def _l2_normalize(embeddings: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / np.clip(norms, 1e-12, None)


def alignment_score(embeddings: np.ndarray, labels: np.ndarray) -> float:
    """Mean squared L2 distance between same-label ("positive") pairs,
    pooled across all label groups. Lower is more aligned."""
    unit = _l2_normalize(embeddings)
    total_sq_dist, total_pairs = 0.0, 0
    for label in np.unique(labels):
        group = unit[labels == label]
        n = len(group)
        if n < 2:
            continue
        group_sum = group.sum(axis=0)
        sum_sq_norm = float(group_sum @ group_sum)
        # mean pairwise cosine similarity of n unit vectors, centroid trick
        mean_cos_sim = (sum_sq_norm - n) / (n * (n - 1))
        mean_sq_dist = 2 - 2 * mean_cos_sim  # ||x-y||^2 = 2 - 2*cos(x,y) for unit vectors
        n_pairs = n * (n - 1) // 2
        total_sq_dist += mean_sq_dist * n_pairs
        total_pairs += n_pairs
    return total_sq_dist / total_pairs


def uniformity_score(
    embeddings: np.ndarray, t: float = 2.0, sample_size: int = 3000, seed: int = 0
) -> float:
    """log-mean-exp(-t * ||x-y||^2) over all pairs in a fixed-seed random
    subsample. Lower (more negative) is more uniform (less collapsed)."""
    unit = _l2_normalize(embeddings)
    if len(unit) > sample_size:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(unit), size=sample_size, replace=False)
        unit = unit[idx]
    sq_dists = pdist(unit, metric="sqeuclidean")
    return float(np.log(np.mean(np.exp(-t * sq_dists))))
