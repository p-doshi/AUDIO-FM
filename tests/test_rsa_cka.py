"""Sanity checks for the RSA/CKA/RDM math against known/synthetic cases."""
from __future__ import annotations

import numpy as np
import pytest

from audio_comp.geometry.cka import linear_cka
from audio_comp.geometry.rdm import compute_rdm
from audio_comp.geometry.rsa import rsa_score


def test_rsa_identical_rdms_is_one():
    rng = np.random.default_rng(0)
    embeddings = rng.normal(size=(20, 8))
    rdm = compute_rdm(embeddings)
    assert np.isclose(rsa_score(rdm, rdm), 1.0)


def test_rsa_independent_random_rdms_near_zero():
    rng = np.random.default_rng(1)
    rdm_a = compute_rdm(rng.normal(size=(50, 16)))
    rdm_b = compute_rdm(rng.normal(size=(50, 16)))
    # not exactly zero with finite samples, but should stay far from 1
    assert abs(rsa_score(rdm_a, rdm_b)) < 0.3


def test_rsa_raises_on_shape_mismatch():
    rdm_a = compute_rdm(np.random.randn(10, 4))
    rdm_b = compute_rdm(np.random.randn(12, 4))
    with pytest.raises(ValueError):
        rsa_score(rdm_a, rdm_b)


def test_cka_identity_is_one():
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 12))
    assert np.isclose(linear_cka(x, x), 1.0, atol=1e-6)


def test_cka_invariant_to_orthogonal_transform():
    rng = np.random.default_rng(3)
    x = rng.normal(size=(30, 12))
    q, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    y = x @ q
    assert np.isclose(linear_cka(x, y), 1.0, atol=1e-6)


def test_cka_raises_on_row_mismatch():
    x = np.random.randn(10, 4)
    y = np.random.randn(12, 4)
    with pytest.raises(ValueError):
        linear_cka(x, y)
