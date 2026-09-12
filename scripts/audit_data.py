#!/usr/bin/env python3
"""Phase 2. Quantify the defects in the raw corpus before any modelling happens.

Answers, with numbers rather than adjectives:
  how much of the text is redaction padding
  how many narratives are exact duplicates
  how many are near duplicates, the templated bulk filings
  how long are narratives, and how many are too short to cluster
  how much non English and non ASCII content is there
  what does the vocabulary look like before cleaning

Writes reports/audit_*.csv and prints a summary. Read only with respect to data/raw.

Usage:
    python scripts/audit_data.py
"""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
REPORTS = ROOT / "reports"

REDACTION = re.compile(r"X{2,}")
WORD = re.compile(r"[a-z']+")

# MinHash settings. 32 permutations banded 8 x 4 finds pairs above roughly 0.6 Jaccard,
# which is the range templated filings sit in once names and amounts are redacted out.
PERMS, BANDS = 32, 8
ROWS_PER_BAND = PERMS // BANDS
SHINGLE = 5
MERSENNE = (1 << 61) - 1


def normalise(text: str) -> str:
    """Lowercase, drop redaction runs and punctuation, collapse whitespace."""
    t = REDACTION.sub(" ", text)
    t = t.lower()
    t = re.sub(r"[^a-z\s']", " ", t)
    return re.sub(r"\s+", " ", t).strip()


def minhash_signatures(docs: list[str], seed: int = 0) -> np.ndarray:
    """Signature matrix, one row per document, PERMS columns."""
    rng = np.random.default_rng(seed)
    a = rng.integers(1, MERSENNE, size=PERMS, dtype=np.uint64)
    b = rng.integers(0, MERSENNE, size=PERMS, dtype=np.uint64)
    sig = np.full((len(docs), PERMS), np.iinfo(np.uint64).max, dtype=np.uint64)

    for i, doc in enumerate(docs):
        words = doc.split()
        if len(words) < SHINGLE:
            shingles = {doc} if doc else set()
        else:
            shingles = {" ".join(words[j:j + SHINGLE]) for j in range(len(words) - SHINGLE + 1)}
        if not shingles:
            continue
        h = np.array([hash(s) & 0xFFFFFFFFFFFFFFF for s in shingles], dtype=np.uint64)
        # (a * h + b) mod Mersenne prime, vectorised over permutations
        vals = (np.outer(h, a) + b) % np.uint64(MERSENNE)
        sig[i] = vals.min(axis=0)
    return sig


def lsh_clusters(sig: np.ndarray) -> dict[int, list[int]]:
    """Union find over documents that collide in any band."""
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
        chunk = sig[:, band * ROWS_PER_BAND:(band + 1) * ROWS_PER_BAND]
        for i, row in enumerate(chunk):
            buckets.setdefault(row.tobytes(), []).append(i)
        for members in buckets.values():
            for m in members[1:]:
                union(members[0], m)

    groups: dict[int, list[int]] = {}
    for i in range(sig.shape[0]):
        groups.setdefault(find(i), []).append(i)
    return {k: v for k, v in groups.items() if len(v) > 1}


def audit(df: pd.DataFrame, name: str) -> dict:
    text = df["complaint_what_happened"].fillna("")
    n = len(df)
    out: dict[str, object] = {"corpus": name, "documents": n}

    # --- redaction padding -------------------------------------------------
    red_counts = text.str.count(REDACTION)
    red_chars = text.str.findall(REDACTION).apply(lambda xs: sum(len(x) for x in xs))
    total_chars = text.str.len()
    out["docs_with_redaction_pct"] = round(100 * (red_counts > 0).mean(), 1)
    out["redaction_runs_per_doc_median"] = float(red_counts.median())
    out["redaction_share_of_chars_pct"] = round(100 * red_chars.sum() / total_chars.sum(), 2)

    # --- length ------------------------------------------------------------
    words = text.apply(lambda t: len(WORD.findall(t.lower())))
    out["chars_median"] = int(total_chars.median())
    out["words_median"] = int(words.median())
    out["words_p05"] = int(words.quantile(0.05))
    out["words_p95"] = int(words.quantile(0.95))
    out["docs_under_20_words_pct"] = round(100 * (words < 20).mean(), 1)

    # --- encoding ----------------------------------------------------------
    non_ascii = text.apply(lambda t: any(ord(c) > 127 for c in t))
    out["docs_with_non_ascii_pct"] = round(100 * non_ascii.mean(), 1)

    # --- duplication -------------------------------------------------------
    norm = text.apply(normalise)
    out["exact_duplicates"] = int(norm.duplicated().sum())
    out["exact_duplicates_pct"] = round(100 * norm.duplicated().mean(), 1)

    sig = minhash_signatures(norm.tolist())
    groups = lsh_clusters(sig)
    near_dupes = sum(len(v) - 1 for v in groups.values())
    out["near_duplicate_groups"] = len(groups)
    out["near_duplicate_surplus"] = near_dupes
    out["near_duplicate_surplus_pct"] = round(100 * near_dupes / n, 1)
    out["largest_template_group"] = max((len(v) for v in groups.values()), default=0)

    # --- vocabulary --------------------------------------------------------
    counter: Counter[str] = Counter()
    for t in norm:
        counter.update(t.split())
    out["vocabulary_raw"] = len(counter)
    out["hapax_pct"] = round(100 * sum(1 for c in counter.values() if c == 1) / len(counter), 1)

    return out, counter, groups, words


def main() -> int:
    REPORTS.mkdir(exist_ok=True)
    rows, extras = [], {}
    for name in ["stratified", "natural"]:
        df = pd.read_parquet(RAW / f"complaints_{name}.parquet")
        summary, counter, groups, words = audit(df, name)
        rows.append(summary)
        extras[name] = (df, counter, groups, words)

    audit_df = pd.DataFrame(rows).set_index("corpus").T
    audit_df.to_csv(REPORTS / "audit_summary.csv")
    print(audit_df.to_string())

    # Top raw tokens, the vocabulary we are about to inherit
    df, counter, groups, words = extras["stratified"]
    top = pd.DataFrame(counter.most_common(30), columns=["token", "count"])
    top["pct_of_tokens"] = (100 * top["count"] / sum(counter.values())).round(2)
    top.to_csv(REPORTS / "audit_top_tokens.csv", index=False)
    print("\nTop 25 tokens in the stratified corpus before stopword removal:")
    print(top.head(25).to_string(index=False))

    # Which products the templated filings concentrate in
    big = sorted(groups.values(), key=len, reverse=True)[:10]
    tpl = pd.DataFrame([{
        "group_size": len(g),
        "dominant_product": df.iloc[g]["product"].mode().iat[0][:45],
        "dominant_issue": df.iloc[g]["issue"].mode().iat[0][:45],
        "sample": df.iloc[g[0]]["complaint_what_happened"][:110].replace("\n", " "),
    } for g in big])
    tpl.to_csv(REPORTS / "audit_template_groups.csv", index=False)
    print("\nTen largest near duplicate groups in the stratified corpus:")
    print(tpl.to_string(index=False))

    # Length by product, to see whether any class is systematically short
    bylen = df.assign(words=words).groupby("product")["words"].agg(["median", "count"])
    bylen = bylen.sort_values("median", ascending=False)
    bylen.to_csv(REPORTS / "audit_length_by_product.csv")
    print("\nMedian narrative length in words, by product:")
    print(bylen.to_string())
    print(f"\nWrote 4 CSVs to {REPORTS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
