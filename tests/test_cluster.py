"""Tests for phase 6. The length variance test is the one that matters."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.cluster import (balance, distinctive_terms, fit_kmeans,
                         length_variance_explained, make_dense_sample, top_terms)


def test_length_variance_explained_detects_a_pure_length_partition():
    """A partition built from length alone must score near 1."""
    lengths = np.concatenate([np.full(50, 10.0), np.full(50, 500.0)])
    labels = np.concatenate([np.zeros(50, int), np.ones(50, int)])
    assert length_variance_explained(labels, lengths) > 0.99


def test_length_variance_explained_ignores_a_length_blind_partition():
    """A partition uncorrelated with length must score near 0."""
    rng = np.random.default_rng(0)
    lengths = rng.normal(100, 30, size=400)
    labels = rng.integers(0, 4, size=400)
    assert length_variance_explained(labels, lengths) < 0.05


def test_length_variance_explained_handles_constant_lengths():
    assert length_variance_explained(np.array([0, 1, 0, 1]), np.array([7.0] * 4)) == 0.0


def test_balance_reports_a_degenerate_split():
    labels = np.concatenate([np.zeros(99, int), np.ones(1, int)])
    b = balance(labels)
    assert b["largest_cluster_pct"] == 99.0
    assert b["size_ratio"] == 99.0
    assert b["normalised_entropy"] < 0.1


def test_balance_reports_an_even_split():
    b = balance(np.repeat([0, 1, 2, 3], 25))
    assert b["size_ratio"] == 1.0
    assert b["normalised_entropy"] == pytest.approx(1.0, abs=0.01)


@pytest.fixture
def blobs():
    """Three well separated term profiles, so clustering has a right answer to find."""
    rng = np.random.default_rng(1)
    vocab = [f"w{i}" for i in range(30)]
    rows = []
    for group in range(3):
        for _ in range(40):
            v = np.zeros(30)
            v[group * 10:(group + 1) * 10] = rng.integers(1, 5, size=10)
            rows.append(v)
    return sparse.csr_matrix(np.array(rows)), vocab


def test_kmeans_recovers_known_groups(blobs):
    X, _ = blobs
    km = fit_kmeans(X, 3)
    # each true group of 40 should land in a single cluster
    for g in range(3):
        assert len(set(km.labels_[g * 40:(g + 1) * 40])) == 1


def test_kmeans_is_reproducible(blobs):
    X, _ = blobs
    assert np.array_equal(fit_kmeans(X, 3).labels_, fit_kmeans(X, 3).labels_)


def test_top_terms_returns_the_right_shape(blobs):
    X, vocab = blobs
    km = fit_kmeans(X, 3)
    terms = top_terms(km, vocab, n=5)
    assert set(terms) == {0, 1, 2}
    assert all(len(v) == 5 for v in terms.values())
    assert all(t in vocab for v in terms.values() for t in v)


def test_distinctive_terms_separate_clusters(blobs):
    """Each blob uses its own ten terms, so the distinctive terms must not overlap."""
    X, vocab = blobs
    km = fit_kmeans(X, 3)
    picked = [set(v) for v in distinctive_terms(km, vocab, n=5).values()]
    assert picked[0].isdisjoint(picked[1])
    assert picked[1].isdisjoint(picked[2])


def test_dense_sample_is_smaller_than_the_matrix(blobs):
    X, _ = blobs
    idx, dense = make_dense_sample(X, size=50)
    assert dense.shape == (50, X.shape[1])
    assert len(set(idx.tolist())) == 50, "sampling must be without replacement"
