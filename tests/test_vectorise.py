"""Tests for phase 5. The leakage test is the important one."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.vectorise import RANDOM_STATE, length_coupling, make_vectoriser, split_corpus

PRODUCTS = ["Credit card", "Mortgage", "Debt collection", "Student loan"]


@pytest.fixture
def corpus():
    rng = np.random.default_rng(0)
    words = {"Credit card": "card charge statement interest limit",
             "Mortgage": "mortgage escrow servicer loan property tax",
             "Debt collection": "collector debt letter validation harassment",
             "Student loan": "student loan forbearance servicer repayment"}
    rows = []
    for i, p in enumerate(PRODUCTS):
        for j in range(60):
            n = rng.integers(6, 20)
            # A unique token per document, so the corpus has a hapax tail like the real one
            # (37.5% of real types occur exactly once) and the min_df gate has something to cut.
            body = " ".join(rng.choice(words[p].split(), n))
            rows.append({"clean_text": f"{body} rare{i}x{j}", "product": p})
    df = pd.DataFrame(rows)
    df["n_tokens"] = df["clean_text"].str.split().str.len()
    return df


def test_split_is_disjoint_and_stratified(corpus):
    train, test = split_corpus(corpus)
    assert len(train) + len(test) == len(corpus)
    # every class present in both halves, in roughly the original proportion
    for p in PRODUCTS:
        assert (train["product"] == p).sum() > 0
        assert (test["product"] == p).sum() > 0
    share_train = train["product"].value_counts(normalize=True).sort_index()
    share_all = corpus["product"].value_counts(normalize=True).sort_index()
    assert np.allclose(share_train.values, share_all.values, atol=0.02)


def test_split_is_reproducible(corpus):
    a, _ = split_corpus(corpus)
    b, _ = split_corpus(corpus)
    assert a.equals(b), f"split must be deterministic at seed {RANDOM_STATE}"


def test_vocabulary_does_not_leak_from_test_into_train(corpus):
    """The vectoriser must be fit on train only. The reference solution fits on everything."""
    train, test = split_corpus(corpus)
    test = test.copy()
    test.loc[test.index[0], "clean_text"] = "zzzunseenterm " * 5

    vec = make_vectoriser("tfidf", min_df=1)
    vec.fit(train["clean_text"])
    assert "zzzunseenterm" not in vec.vocabulary_, "a test only term reached the vocabulary"

    Xte = vec.transform(test["clean_text"])
    assert Xte.shape[1] == len(vec.vocabulary_), "transform must not widen the matrix"


def test_tfidf_rows_are_l2_normalised(corpus):
    vec = make_vectoriser("tfidf", min_df=1)
    X = vec.fit_transform(corpus["clean_text"])
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    assert np.allclose(norms, 1.0), "L2 normalisation is what removes the length effect"


def test_count_rows_are_not_normalised(corpus):
    vec = make_vectoriser("count", min_df=1)
    X = vec.fit_transform(corpus["clean_text"])
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    assert norms.std() > 0, "raw counts must vary with document length, that is the confound"


def test_length_coupling_separates_the_two_representations(corpus):
    """F2, stated as a test: counts should encode length, TF IDF should not."""
    lengths = corpus["n_tokens"].to_numpy()
    Xc = make_vectoriser("count", min_df=1).fit_transform(corpus["clean_text"])
    Xt = make_vectoriser("tfidf", min_df=1).fit_transform(corpus["clean_text"])

    c = length_coupling(Xc, lengths, sample=200)
    t = length_coupling(Xt, lengths, sample=200)

    assert c["row_norm_corr_with_length"] > 0.8
    assert t["row_norm_std"] == 0.0
    assert c["pair_distance_corr_with_length_gap"] > t["pair_distance_corr_with_length_gap"]


def test_gates_are_applied(corpus):
    wide = make_vectoriser("count", min_df=1).fit(corpus["clean_text"])
    narrow = make_vectoriser("count", min_df=50).fit(corpus["clean_text"])
    assert len(narrow.vocabulary_) < len(wide.vocabulary_)


def test_unknown_kind_is_rejected():
    with pytest.raises(ValueError):
        make_vectoriser("word2vec")
