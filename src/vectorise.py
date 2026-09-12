"""Phase 5. Turn the cleaned corpus into document term matrices.

Two representations, built on identical vocabulary gates so they are directly comparable:
raw counts as the reference solution uses, and L2 normalised TF IDF. The brief asks for both.
Phase 3 finding F2 turned that into a testable hypothesis rather than a box to tick: narrative
length varies by a factor of 2.2 across products, raw counts scale with length, so KMeans under
Euclidean distance can separate categories on length alone. TF IDF should remove that effect.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42
TEST_SIZE = 0.2


def split_corpus(df: pd.DataFrame, text_col: str = "clean_text",
                 label_col: str = "product") -> tuple[pd.DataFrame, pd.DataFrame]:
    """Stratified train and test split.

    The vectoriser and the clustering are fit on train only. The reference solution fits on the
    entire corpus, which makes it impossible to say whether anything generalises.
    """
    train, test = train_test_split(df, test_size=TEST_SIZE, random_state=RANDOM_STATE,
                                   stratify=df[label_col])
    return train.reset_index(drop=True), test.reset_index(drop=True)


def make_vectoriser(kind: str, min_df: int = 5, max_df: float = 1.0,
                    ngram: tuple[int, int] = (1, 1)):
    """CountVectorizer or TfidfVectorizer with identical vocabulary gates.

    Note on max_df. The reference solution uses max_df=0.7. On this corpus that threshold
    removes exactly zero terms, because the most widespread term after stopword removal
    ("account") appears in only 52.2 percent of documents. The parameter is inert here and is
    kept at 1.0 by default rather than copied across for the look of it. See
    reports/vectoriser_comparison.csv.
    """
    common = {"min_df": min_df, "max_df": max_df, "ngram_range": ngram, "lowercase": False}
    if kind == "count":
        return CountVectorizer(**common)
    if kind == "tfidf":
        return TfidfVectorizer(sublinear_tf=True, norm="l2", **common)
    raise ValueError(f"unknown vectoriser kind: {kind}")


def describe(X: sparse.csr_matrix, name: str, vocab_size: int, fitted_on: int) -> dict:
    density = X.nnz / (X.shape[0] * X.shape[1])
    return {
        "representation": name,
        "documents": X.shape[0],
        "features": X.shape[1],
        "vocabulary": vocab_size,
        "stored_values": X.nnz,
        "density_pct": round(100 * density, 4),
        "sparsity_pct": round(100 * (1 - density), 4),
        "nonzero_per_doc_median": int(np.median(np.diff(X.indptr))),
        "memory_mb": round((X.data.nbytes + X.indices.nbytes + X.indptr.nbytes) / 1e6, 1),
        "fitted_on_documents": fitted_on,
    }


def length_coupling(X: sparse.csr_matrix, lengths: np.ndarray, sample: int = 3000,
                    seed: int = 0) -> dict:
    """How strongly does this representation encode document length rather than content?

    Two measures:
      row_norm_corr      correlation between a document's vector norm and its token count
      pair_distance_corr correlation between pairwise Euclidean distance and the difference in
                         length, over a random sample of document pairs

    If the second number is high, KMeans under Euclidean distance is partly clustering on
    length. That is the F2 hypothesis, stated so it can be falsified.
    """
    rng = np.random.default_rng(seed)
    norms = np.sqrt(X.multiply(X).sum(axis=1)).A.ravel()
    row_corr = float(np.corrcoef(norms, lengths)[0, 1]) if norms.std() > 1e-12 else 0.0

    n = X.shape[0]
    idx = rng.choice(n, size=min(sample, n), replace=False)
    a, b = idx[: len(idx) // 2], idx[len(idx) // 2:]
    m = min(len(a), len(b))
    a, b = a[:m], b[:m]
    diff = X[a] - X[b]
    dist = np.sqrt(diff.multiply(diff).sum(axis=1)).A.ravel()
    len_gap = np.abs(lengths[a] - lengths[b])
    pair_corr = float(np.corrcoef(dist, len_gap)[0, 1])

    return {"row_norm_corr_with_length": round(row_corr, 3),
            "pair_distance_corr_with_length_gap": round(pair_corr, 3),
            "row_norm_std": round(float(norms.std()), 4)}
