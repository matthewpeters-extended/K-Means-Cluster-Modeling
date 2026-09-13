"""Phase 7. Score the clusters against the labels the bureau assigned by hand.

This is the phase the reference solution has no equivalent of, and it is the reason this corpus
was chosen over the original tweet dataset. Phase 6 showed that internal metrics recommend the
broken model. Only ground truth can settle it.

Every headline number is reported next to two baselines:
  majority   always predict the most common product. Purity cannot be called good until it
             beats this, and on an imbalanced corpus it often does not.
  permuted   the same cluster sizes, assigned at random. This is what the metric scores when
             the clustering carries no information at all, which for purity is far above zero.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import (adjusted_rand_score, completeness_score,
                             homogeneity_score, normalized_mutual_info_score)


def purity(labels: np.ndarray, truth: np.ndarray) -> float:
    """Share of documents in the majority class of their own cluster.

    Not corrected for chance, which is precisely why it needs the permuted baseline beside it.
    """
    total = 0
    for c in np.unique(labels):
        classes, counts = np.unique(truth[labels == c], return_counts=True)
        total += counts.max()
    return float(total / len(labels))


def permuted_baseline(labels: np.ndarray, truth: np.ndarray, repeats: int = 20,
                      seed: int = 0) -> dict:
    """What each metric scores when cluster sizes are kept but membership is shuffled."""
    rng = np.random.default_rng(seed)
    pur, ari, nmi = [], [], []
    for _ in range(repeats):
        shuffled = rng.permutation(labels)
        pur.append(purity(shuffled, truth))
        ari.append(adjusted_rand_score(truth, shuffled))
        nmi.append(normalized_mutual_info_score(truth, shuffled))
    return {"purity": float(np.mean(pur)), "purity_sd": float(np.std(pur)),
            "ari": float(np.mean(ari)), "nmi": float(np.mean(nmi))}


def majority_baseline(truth: np.ndarray) -> float:
    _, counts = np.unique(truth, return_counts=True)
    return float(counts.max() / len(truth))


def score(labels: np.ndarray, truth: np.ndarray, seed: int = 0) -> dict:
    """Every metric for one clustering, with both baselines and the lift over each."""
    base = permuted_baseline(labels, truth, seed=seed)
    maj = majority_baseline(truth)
    pur = purity(labels, truth)
    return {
        "purity": round(pur, 4),
        "purity_permuted": round(base["purity"], 4),
        "purity_majority": round(maj, 4),
        "purity_lift_over_permuted": round(pur - base["purity"], 4),
        "beats_majority": bool(pur > maj),
        "ari": round(adjusted_rand_score(truth, labels), 4),
        "ari_permuted": round(base["ari"], 4),
        "nmi": round(normalized_mutual_info_score(truth, labels), 4),
        "nmi_permuted": round(base["nmi"], 4),
        "homogeneity": round(homogeneity_score(truth, labels), 4),
        "completeness": round(completeness_score(truth, labels), 4),
        "clusters": int(len(np.unique(labels))),
        "classes": int(len(np.unique(truth))),
    }


def contingency(labels: np.ndarray, truth: np.ndarray) -> pd.DataFrame:
    """Cluster by class counts, the table behind every summary number."""
    return pd.crosstab(pd.Series(labels, name="cluster"), pd.Series(truth, name="label"))


def company_tokens(companies: pd.Series) -> set[str]:
    """Vocabulary drawn from company names, for measuring how much of the clustering is
    really firm recognition rather than complaint semantics."""
    generic = {"inc", "llc", "co", "company", "corp", "corporation", "the", "and", "of",
               "bank", "financial", "services", "group", "holdings", "na", "usa", "national",
               "credit", "card", "loan", "mortgage", "america", "american", "first", "us"}
    out: set[str] = set()
    for name in companies.dropna().unique():
        for tok in str(name).lower().replace(",", " ").replace(".", " ").split():
            tok = "".join(ch for ch in tok if ch.isalpha())
            if len(tok) >= 3 and tok not in generic:
                out.add(tok)
    return out


def brand_share(distinctive: dict[int, list[str]], brands: set[str]) -> dict:
    """Fraction of each cluster's distinctive terms that are company names."""
    per_cluster = {c: sum(1 for t in terms if t in brands) / len(terms)
                   for c, terms in distinctive.items()}
    return {"per_cluster": {c: round(v, 3) for c, v in per_cluster.items()},
            "overall": round(float(np.mean(list(per_cluster.values()))), 3)}
