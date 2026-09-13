"""Phase 8. The two topic models KMeans is usually compared against.

The reference solution argues for KMeans over LDA and NMF on grounds of simplicity, speed,
hard assignment and interpretability, but never fits either alternative. Here they are fitted
and scored identically, so the argument can be settled rather than asserted.

One representation decision matters and is not arbitrary:

  NMF runs on TF IDF. Frobenius loss NMF is a matrix factorisation and works on any nonnegative
  matrix; TF IDF is the standard choice and keeps it comparable with the best KMeans setup.

  LDA runs on raw counts, because it has to. LDA is a generative model of how integer word
  counts arise, and TF IDF values are neither integers nor counts.

That gives a clean secondary test. Phase 6 showed KMeans on raw counts collapses into length
bands. LDA normalises each document through its Dirichlet topic distribution, so it should NOT
suffer the same failure despite consuming the same matrix. If that holds, the phase 6 defect
was KMeans under Euclidean distance rather than raw counts as such.
"""

from __future__ import annotations

import numpy as np
from sklearn.decomposition import NMF, LatentDirichletAllocation

RANDOM_STATE = 42
LDA_MAX_ITER = 25


def fit_nmf(X, k: int, seed: int = RANDOM_STATE) -> NMF:
    return NMF(n_components=k, init="nndsvd", random_state=seed,
               max_iter=400, tol=1e-4).fit(X)


def fit_lda(X, k: int, seed: int = RANDOM_STATE) -> LatentDirichletAllocation:
    return LatentDirichletAllocation(
        n_components=k, max_iter=LDA_MAX_ITER, learning_method="online",
        batch_size=512, random_state=seed, n_jobs=-1).fit(X)


def hard_assign(doc_topic: np.ndarray) -> np.ndarray:
    """Collapse a soft topic distribution to one label per document.

    This is the comparison the reference solution's argument turns on: it claims hard
    assignment makes KMeans easier for a business to act on. Taking the argmax gives the
    probabilistic models the same output format, so the claim can be tested on equal terms.
    """
    return doc_topic.argmax(axis=1)


def assignment_confidence(doc_topic: np.ndarray) -> dict:
    """How decisive the soft assignment is.

    A model whose top topic carries 0.95 of the mass is effectively making a hard assignment
    anyway. One sitting near 1/k is hedging, and the argmax label is close to arbitrary.
    """
    norm = doc_topic / np.maximum(doc_topic.sum(axis=1, keepdims=True), 1e-12)
    top = norm.max(axis=1)
    k = doc_topic.shape[1]
    return {"mean_top_weight": round(float(top.mean()), 4),
            "median_top_weight": round(float(np.median(top)), 4),
            "uniform_weight": round(1 / k, 4),
            "share_above_half": round(float((top > 0.5).mean()), 4)}


def topic_terms(components: np.ndarray, vocab: list[str], n: int = 12) -> dict[int, list[str]]:
    order = components.argsort(axis=1)[:, ::-1]
    return {t: [vocab[i] for i in order[t, :n]] for t in range(components.shape[0])}


def distinctive_topic_terms(components: np.ndarray, vocab: list[str],
                            n: int = 12) -> dict[int, list[str]]:
    """Terms characteristic of a topic rather than merely frequent within it."""
    norm = components / np.maximum(components.sum(axis=1, keepdims=True), 1e-12)
    contrast = norm - norm.mean(axis=0, keepdims=True)
    order = contrast.argsort(axis=1)[:, ::-1]
    return {t: [vocab[i] for i in order[t, :n]] for t in range(components.shape[0])}
