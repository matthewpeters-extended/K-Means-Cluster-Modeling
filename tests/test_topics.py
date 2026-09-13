"""Tests for phase 8."""

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.topics import (assignment_confidence, distinctive_topic_terms, fit_lda,
                        fit_nmf, hard_assign, topic_terms)


@pytest.fixture
def blobs():
    """Three disjoint term profiles, so a topic model has a right answer to find."""
    rng = np.random.default_rng(1)
    vocab = [f"w{i}" for i in range(30)]
    rows = []
    for group in range(3):
        for _ in range(40):
            v = np.zeros(30)
            v[group * 10:(group + 1) * 10] = rng.integers(1, 6, size=10)
            rows.append(v)
    return sparse.csr_matrix(np.array(rows, dtype=float)), vocab


def test_hard_assign_takes_the_argmax():
    soft = np.array([[0.1, 0.7, 0.2], [0.6, 0.3, 0.1], [0.2, 0.2, 0.6]])
    assert np.array_equal(hard_assign(soft), np.array([1, 0, 2]))


def test_assignment_confidence_detects_a_decisive_model():
    soft = np.array([[0.95, 0.03, 0.02]] * 50)
    c = assignment_confidence(soft)
    assert c["mean_top_weight"] == pytest.approx(0.95, abs=0.01)
    assert c["share_above_half"] == 1.0


def test_assignment_confidence_detects_a_hedging_model():
    """A model sitting at uniform is not really assigning anything."""
    soft = np.full((50, 4), 0.25)
    c = assignment_confidence(soft)
    assert c["mean_top_weight"] == pytest.approx(c["uniform_weight"], abs=0.01)
    assert c["share_above_half"] == 0.0


def test_confidence_is_invariant_to_unnormalised_input():
    """NMF weights do not sum to one, so the measure must normalise first."""
    a = assignment_confidence(np.array([[0.8, 0.2]] * 10))
    b = assignment_confidence(np.array([[80.0, 20.0]] * 10))
    assert a["mean_top_weight"] == pytest.approx(b["mean_top_weight"])


def test_nmf_recovers_known_groups(blobs):
    X, _ = blobs
    labels = hard_assign(fit_nmf(X, 3).transform(X))
    for g in range(3):
        assert len(set(labels[g * 40:(g + 1) * 40])) == 1, "each blob must land in one topic"


def test_lda_recovers_known_groups(blobs):
    X, _ = blobs
    labels = hard_assign(fit_lda(X, 3).transform(X))
    for g in range(3):
        assert len(set(labels[g * 40:(g + 1) * 40])) == 1


def test_nmf_is_reproducible(blobs):
    X, _ = blobs
    a = hard_assign(fit_nmf(X, 3).transform(X))
    b = hard_assign(fit_nmf(X, 3).transform(X))
    assert np.array_equal(a, b)


def test_topic_terms_shape(blobs):
    X, vocab = blobs
    terms = topic_terms(fit_nmf(X, 3).components_, vocab, n=5)
    assert set(terms) == {0, 1, 2}
    assert all(len(v) == 5 and all(t in vocab for t in v) for v in terms.values())


def test_distinctive_topic_terms_do_not_overlap(blobs):
    """Each blob owns ten exclusive terms, so distinctive terms must be disjoint."""
    X, vocab = blobs
    picked = [set(v) for v in distinctive_topic_terms(fit_nmf(X, 3).components_, vocab, n=5).values()]
    assert picked[0].isdisjoint(picked[1])
    assert picked[1].isdisjoint(picked[2])
    assert picked[0].isdisjoint(picked[2])
