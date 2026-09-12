"""Phase 6. KMeans over the document term matrices, with the diagnostics the brief skips.

The brief asks for two centroid counts, K = 2 and K = 8, compared and interpreted. Those are
delivered. On top of that we run the sweep the reference solution never ran, K = 2 to 20 scored
on four internal metrics, so the choice of K is answerable from evidence rather than by eye.

Phase 5 finding V2 left one specific thing to falsify: raw counts encode document length, so a
KMeans fit on them may be partitioning by length rather than by topic. `length_variance_explained`
is the test.
"""

from __future__ import annotations

import numpy as np
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.metrics import (calinski_harabasz_score, davies_bouldin_score,
                             silhouette_score)

RANDOM_STATE = 42
N_INIT = 10
SILHOUETTE_SAMPLE = 3000
DENSE_SAMPLE = 4000


def fit_kmeans(X, k: int, seed: int = RANDOM_STATE) -> KMeans:
    return KMeans(n_clusters=k, n_init=N_INIT, random_state=seed, max_iter=300).fit(X)


def length_variance_explained(labels: np.ndarray, lengths: np.ndarray) -> float:
    """Eta squared: the share of document length variance explained by cluster membership.

    This is the falsification test for V2. A partition that is really about topic should
    explain very little of the variance in how long documents are. A partition that is secretly
    a set of length bands will explain a great deal of it.
    """
    grand = lengths.mean()
    total_ss = float(((lengths - grand) ** 2).sum())
    if total_ss == 0:
        return 0.0
    between = 0.0
    for c in np.unique(labels):
        group = lengths[labels == c]
        between += len(group) * (group.mean() - grand) ** 2
    return float(between / total_ss)


def balance(labels: np.ndarray) -> dict:
    """How evenly documents are spread across clusters."""
    counts = np.bincount(labels)
    share = counts / counts.sum()
    entropy = float(-(share * np.log(share + 1e-12)).sum())
    return {"largest_cluster_pct": round(100 * share.max(), 1),
            "smallest_cluster_pct": round(100 * share.min(), 1),
            "size_ratio": round(float(counts.max() / max(counts.min(), 1)), 1),
            "normalised_entropy": round(entropy / np.log(len(counts)), 3)}


def internal_metrics(X, labels: np.ndarray, dense_idx: np.ndarray,
                     dense_X: np.ndarray) -> dict:
    """Silhouette on a sparse sample; Davies Bouldin and Calinski Harabasz on a dense sample.

    DB and CH require dense input. Densifying 14,332 x 6,885 would cost 790 MB, so both are
    computed on a fixed random sample of 4,000 documents, identical across every value of K so
    the comparison between K values stays fair.
    """
    out = {}
    try:
        out["silhouette"] = round(float(silhouette_score(
            X, labels, sample_size=SILHOUETTE_SAMPLE, random_state=RANDOM_STATE)), 4)
    except ValueError:
        out["silhouette"] = np.nan

    sub = labels[dense_idx]
    if len(np.unique(sub)) > 1:
        out["davies_bouldin"] = round(float(davies_bouldin_score(dense_X, sub)), 3)
        out["calinski_harabasz"] = round(float(calinski_harabasz_score(dense_X, sub)), 1)
    else:
        out["davies_bouldin"] = np.nan
        out["calinski_harabasz"] = np.nan
    return out


def top_terms(model: KMeans, vocab: list[str], n: int = 12) -> dict[int, list[str]]:
    """Highest weighted terms in each centroid, which is how a cluster gets its name."""
    order = model.cluster_centers_.argsort()[:, ::-1]
    return {c: [vocab[i] for i in order[c, :n]] for c in range(model.n_clusters)}


def distinctive_terms(model: KMeans, vocab: list[str], n: int = 12) -> dict[int, list[str]]:
    """Terms that separate a cluster from the others, rather than terms that are merely frequent.

    Centroid weight alone surfaces the same common words in every cluster. Subtracting the mean
    centroid shows what is actually characteristic of each one. Word clouds show the first kind
    and hide the second, which is one reason this project reports both.
    """
    centres = model.cluster_centers_
    contrast = centres - centres.mean(axis=0, keepdims=True)
    order = contrast.argsort()[:, ::-1]
    return {c: [vocab[i] for i in order[c, :n]] for c in range(model.n_clusters)}


def make_dense_sample(X, seed: int = 0, size: int = DENSE_SAMPLE) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n = X.shape[0]
    idx = rng.choice(n, size=min(size, n), replace=False)
    block = X[idx]
    dense = np.asarray(block.todense(), dtype=np.float32) if sparse.issparse(block) \
        else np.asarray(block, dtype=np.float32)
    return idx, dense
