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
view structure. `alignment_score()` below uses same-category clips as a
positive-pair proxy — a reasonable stand-in for "should be close
together" but not the literal original definition. **That proxy turned
out to have a real problem, found 2026-08-10**: alignment and uniformity
came back perfectly rank-correlated (Spearman -1.0) across all 9 models
tested, meaning the category proxy wasn't measuring anything independent
of overall embedding-space dispersion — every same-category pair (e.g.
every "ship" clip paired with every other "ship" clip) conflates "close
because genuinely similar" with "close because same broad label."
`alignment_score_paired()` fixes this using true instance-level positive
pairs (original clip vs. a pitch-shifted augmented view of the *same*
clip, built by `audio_comp/data/build_augmented_probe_subset.py`) — the
literal original definition, not an adaptation. Prefer this one; the
category-proxy version is kept for the historical comparison, not as
the recommended metric going forward.

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


def alignment_score_paired(embeddings_a: np.ndarray, embeddings_b: np.ndarray) -> float:
    """Mean squared L2 distance between true instance-level positive
    pairs: embeddings_a[i] and embeddings_b[i] must be (original,
    augmented-view) embeddings of the *same* underlying clip, same
    ordering in both arrays. This is the literal Wang & Isola definition
    (no category-proxy adaptation) -- prefer this over alignment_score()
    when paired-augmentation embeddings are available."""
    if embeddings_a.shape != embeddings_b.shape:
        raise ValueError(f"shape mismatch: {embeddings_a.shape} vs {embeddings_b.shape}")
    a, b = _l2_normalize(embeddings_a), _l2_normalize(embeddings_b)
    sq_dists = np.sum((a - b) ** 2, axis=1)
    return float(np.mean(sq_dists))


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
