"""Tests for phase 7. Baselines are the point, so most of these test the baselines."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.evaluate import (brand_share, company_tokens, contingency, majority_baseline,
                          permuted_baseline, purity, score)


def test_purity_is_one_for_a_perfect_clustering():
    truth = np.array(["a"] * 30 + ["b"] * 30)
    assert purity(truth.copy(), truth) == 1.0


def test_purity_of_a_single_cluster_equals_the_majority_rate():
    """One cluster containing everything scores exactly the majority class rate."""
    truth = np.array(["a"] * 70 + ["b"] * 30)
    labels = np.zeros(100, dtype=int)
    assert purity(labels, truth) == pytest.approx(0.70)
    assert purity(labels, truth) == pytest.approx(majority_baseline(truth))


def test_purity_rises_with_more_clusters_which_is_why_it_needs_a_control():
    """Purity is not chance corrected: give every document its own cluster and it hits 1.0."""
    rng = np.random.default_rng(0)
    truth = rng.choice(["a", "b", "c"], size=90)
    assert purity(np.arange(90), truth) == 1.0


def test_permuted_baseline_kills_ari_but_not_purity():
    rng = np.random.default_rng(1)
    truth = rng.choice(["a", "b", "c"], size=300, p=[0.7, 0.2, 0.1])
    labels = rng.integers(0, 6, size=300)
    base = permuted_baseline(labels, truth, repeats=10)
    assert abs(base["ari"]) < 0.02, "a shuffled labelling must score near zero ARI"
    assert base["purity"] > 0.6, "but its purity stays near the majority rate, not near zero"


def test_majority_baseline():
    assert majority_baseline(np.array(["x"] * 8 + ["y"] * 2)) == pytest.approx(0.8)


def test_score_flags_a_clustering_that_fails_to_beat_majority():
    truth = np.array(["a"] * 90 + ["b"] * 10)
    useless = np.zeros(100, dtype=int)
    s = score(useless, truth)
    assert s["beats_majority"] is False
    assert s["purity"] == pytest.approx(s["purity_majority"])


def test_score_credits_a_clustering_that_does_beat_majority():
    truth = np.array(["a"] * 50 + ["b"] * 50)
    good = np.array([0] * 50 + [1] * 50)
    s = score(good, truth)
    assert s["beats_majority"] is True
    assert s["purity"] == 1.0
    assert s["ari"] == pytest.approx(1.0)
    assert s["purity_lift_over_permuted"] > 0.4


def test_contingency_counts_add_up():
    labels = np.array([0, 0, 1, 1, 1])
    truth = np.array(["a", "b", "b", "b", "c"])
    ct = contingency(labels, truth)
    assert ct.values.sum() == 5
    assert ct.loc[0, "a"] == 1 and ct.loc[0, "b"] == 1
    assert ct.loc[1, "b"] == 2 and ct.loc[1, "c"] == 1


def test_company_tokens_drops_generic_words():
    s = pd.Series(["MOHELA, Inc.", "Wells Fargo Bank NA", "American Express Company"])
    toks = company_tokens(s)
    assert "mohela" in toks and "fargo" in toks and "express" in toks
    for generic in ["inc", "bank", "company", "american", "na"]:
        assert generic not in toks


def test_brand_share_counts_company_terms():
    brands = {"mohela", "carvana"}
    distinctive = {0: ["mohela", "loan", "student", "carvana"], 1: ["payment", "late", "fee", "due"]}
    bs = brand_share(distinctive, brands)
    assert bs["per_cluster"][0] == 0.5
    assert bs["per_cluster"][1] == 0.0
    assert bs["overall"] == 0.25
