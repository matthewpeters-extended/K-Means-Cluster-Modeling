"""Near duplicate detection over text, using MinHash with LSH banding.

Templated bulk filings are the dominant defect in this corpus: measured at 55 percent surplus
in the natural sample, with one template group holding 1,167 documents. Exact string matching
catches under half of it because the templates are paraphrases of one another. See
docs/data_defects.md D2.
"""

from __future__ import annotations

import hashlib
import re

import numpy as np
import pandas as pd

REDACTION = re.compile(r"X{2,}")

# 32 permutations banded 8 x 4 finds pairs above roughly 0.6 Jaccard, which is where
# templated filings sit once names and amounts are redacted out.
PERMS, BANDS, SHINGLE = 32, 8, 5
ROWS_PER_BAND = PERMS // BANDS
MERSENNE = (1 << 61) - 1


def normalise(text: str) -> str:
    """Aggressive normalisation used only for comparing documents, never for modelling."""
    t = REDACTION.sub(" ", text)
    t = t.lower()
    t = re.sub(r"[^a-z\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def shingles(doc: str, k: int = SHINGLE) -> set[str]:
    words = doc.split()
    if len(words) < k:
        return {doc} if doc else set()
    return {" ".join(words[i:i + k]) for i in range(len(words) - k + 1)}


def stable_hash(s: str) -> int:
    """Reproducible 60 bit hash.

    Python's built in hash() is salted per process, so using it here would make the
    deduplication non reproducible between runs of the same script on the same data. That is
    unacceptable in a pipeline whose entire claim is reproducibility.
    """
    return int.from_bytes(hashlib.blake2b(s.encode("utf8"), digest_size=8).digest(),
                          "big") & 0xFFFFFFFFFFFFFFF


def minhash_signatures(docs: list[str], seed: int = 0) -> np.ndarray:
    """One signature row per document, PERMS columns. Deterministic for a given seed."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MERSENNE, size=PERMS, dtype=np.uint64)
    b = rng.integers(0, MERSENNE, size=PERMS, dtype=np.uint64)
    sig = np.full((len(docs), PERMS), np.iinfo(np.uint64).max, dtype=np.uint64)

    for i, doc in enumerate(docs):
        sh = shingles(doc)
        if not sh:
            continue
        h = np.array([stable_hash(s) for s in sh], dtype=np.uint64)
        sig[i] = ((np.outer(h, a) + b) % np.uint64(MERSENNE)).min(axis=0)
    return sig


def lsh_groups(sig: np.ndarray) -> list[list[int]]:
    """Union find over documents colliding in any band. Returns groups of size two or more."""
    parent = list(range(sig.shape[0]))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[max(rx, ry)] = min(rx, ry)

    for band in range(BANDS):
        buckets: dict[bytes, list[int]] = {}
        for i, row in enumerate(sig[:, band * ROWS_PER_BAND:(band + 1) * ROWS_PER_BAND]):
            buckets.setdefault(row.tobytes(), []).append(i)
        for members in buckets.values():
            for m in members[1:]:
                union(members[0], m)

    groups: dict[int, list[int]] = {}
    for i in range(sig.shape[0]):
        groups.setdefault(find(i), []).append(i)
    return [g for g in groups.values() if len(g) > 1]


def collapse_near_duplicates(df: pd.DataFrame, text_col: str,
                             seed: int = 0) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Keep one representative per near duplicate group, recording how many it stood for.

    Returns (collapsed frame with a template_size column, frame of the groups found).
    The longest document in each group is kept, on the reasoning that the fullest version of a
    template carries the most vocabulary. Volume is preserved as a column rather than thrown
    away, because how often a template is filed is real information about consumer behaviour.
    It simply must not vote repeatedly when the centroids are computed.
    """
    norm = df[text_col].fillna("").map(normalise)
    sig = minhash_signatures(norm.tolist(), seed=seed)
    groups = lsh_groups(sig)

    sizes = np.ones(len(df), dtype=int)
    drop: set[int] = set()
    rows = []
    lengths = df[text_col].fillna("").str.len().to_numpy()

    for g in groups:
        keep = max(g, key=lambda i: lengths[i])
        sizes[keep] = len(g)
        drop.update(set(g) - {keep})
        rows.append({"group_size": len(g), "kept_index": keep,
                     "product": df.iloc[keep].get("product")})

    out = df.copy()
    out["template_size"] = sizes
    out = out.drop(out.index[sorted(drop)])
    return out.reset_index(drop=True), pd.DataFrame(rows)
