#!/usr/bin/env python3
"""Phase 5. Build and compare document term matrices, then save the ones phase 6 will cluster.

Produces:
  reports/vectoriser_gate_sweep.csv    vocabulary retention across min_df and max_df
  reports/vectoriser_comparison.csv    shape, sparsity and memory for each representation
  reports/length_coupling.csv          how much each representation encodes length, not content
  data/processed/features/*.npz        the fitted matrices, train and test
  reports/figures/07_vectorisation.png

Usage:
    python scripts/build_features.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.feature_extraction.text import CountVectorizer  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.vectorise import (RANDOM_STATE, describe, length_coupling,  # noqa: E402
                           make_vectoriser, split_corpus)

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
FEATURES = PROCESSED / "features"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

INK, GRID, BLUE, ORANGE, TEAL = "#1b1b1b", "#d9d9d9", "#2b6cb0", "#dd6b20", "#2c7a7b"

# Representations to build. Same gates throughout so the comparison is like for like.
CONFIGS = [
    ("count_unigram", "count", 5, 1.0, (1, 1)),
    ("tfidf_unigram", "tfidf", 5, 1.0, (1, 1)),
    ("tfidf_bigram", "tfidf", 5, 1.0, (1, 2)),
    ("count_unigram_maxdf07", "count", 5, 0.7, (1, 1)),   # the reference solution's gate
]


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def gate_sweep(texts: list[str]) -> pd.DataFrame:
    """How much vocabulary and how many tokens survive each gate, measured not assumed."""
    cv = CountVectorizer(min_df=1, max_df=1.0, lowercase=False)
    X = cv.fit_transform(texts)
    vocab = np.array(cv.get_feature_names_out())
    df_counts = np.asarray((X > 0).sum(axis=0)).ravel()
    n, total_tokens = X.shape[0], X.sum()

    rows = []
    for m in [1, 2, 3, 5, 10, 20, 50]:
        keep = df_counts >= m
        rows.append({"gate": "min_df", "value": m, "vocabulary": int(keep.sum()),
                     "vocabulary_pct": round(100 * keep.mean(), 1),
                     "tokens_retained_pct": round(100 * X[:, keep].sum() / total_tokens, 2),
                     "terms_removed": ""})
    for mx in [1.0, 0.9, 0.7, 0.5, 0.3, 0.2]:
        drop = df_counts > mx * n
        removed = vocab[drop][np.argsort(-df_counts[drop])][:8]
        rows.append({"gate": "max_df", "value": mx,
                     "vocabulary": int((~drop).sum()),
                     "vocabulary_pct": round(100 * (~drop).mean(), 1),
                     "tokens_retained_pct": round(100 * X[:, ~drop].sum() / total_tokens, 2),
                     "terms_removed": ", ".join(removed)})
    return pd.DataFrame(rows), df_counts, vocab, n


def main() -> int:
    src = PROCESSED / "corpus_stratified.parquet"
    if not src.exists():
        print("Run scripts/build_corpus.py first.", file=sys.stderr)
        return 1
    FEATURES.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(src)
    train, test = split_corpus(df)
    print(f"corpus {len(df):,}  ->  train {len(train):,}  test {len(test):,} "
          f"(stratified by product, seed {RANDOM_STATE})")

    # --- gates, chosen from the data ---------------------------------------
    sweep, df_counts, vocab, n = gate_sweep(train["clean_text"].tolist())
    sweep.to_csv(REPORTS / "vectoriser_gate_sweep.csv", index=False)
    print("\n--- vocabulary gates on the training split ---")
    print(sweep.to_string(index=False))

    # --- build each representation -----------------------------------------
    rows, coupling_rows = [], []
    train_len = train["n_tokens"].to_numpy()
    for name, kind, min_df, max_df, ngram in CONFIGS:
        vec = make_vectoriser(kind, min_df=min_df, max_df=max_df, ngram=ngram)
        Xtr = vec.fit_transform(train["clean_text"])
        Xte = vec.transform(test["clean_text"])

        info = describe(Xtr, name, len(vec.vocabulary_), len(train))
        info.update({"kind": kind, "min_df": min_df, "max_df": max_df,
                     "ngram": f"{ngram[0]} to {ngram[1]}", "test_documents": Xte.shape[0]})
        rows.append(info)

        c = length_coupling(Xtr, train_len)
        c["representation"] = name
        coupling_rows.append(c)

        sparse.save_npz(FEATURES / f"{name}_train.npz", Xtr)
        sparse.save_npz(FEATURES / f"{name}_test.npz", Xte)
        (FEATURES / f"{name}_vocab.json").write_text(
            json.dumps(sorted(vec.vocabulary_, key=vec.vocabulary_.get)))
        print(f"  built {name:<24} {Xtr.shape[0]:>6} x {Xtr.shape[1]:<7} "
              f"sparsity {info['sparsity_pct']:.3f}%")

    comp = pd.DataFrame(rows)
    comp.to_csv(REPORTS / "vectoriser_comparison.csv", index=False)
    coup = pd.DataFrame(coupling_rows).set_index("representation")
    coup.to_csv(REPORTS / "length_coupling.csv")

    print("\n--- representations ---")
    print(comp[["representation", "documents", "features", "sparsity_pct",
                "nonzero_per_doc_median", "memory_mb"]].to_string(index=False))
    print("\n--- length coupling: does the representation encode length rather than content? ---")
    print(coup.to_string())

    # --- save the split so phase 6 uses exactly these rows ------------------
    train.to_parquet(FEATURES / "train.parquet", index=False)
    test.to_parquet(FEATURES / "test.parquet", index=False)

    # --- figure -------------------------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))

    s = sweep[sweep.gate == "min_df"]
    axes[0].plot(s.value, s.vocabulary_pct, marker="o", color=BLUE, label="vocabulary kept")
    axes[0].plot(s.value, s.tokens_retained_pct, marker="s", color=ORANGE, label="tokens kept")
    axes[0].set_xscale("log")
    style(axes[0], "min_df: cheap to cut the vocabulary, costly to cut tokens",
          "min_df (documents)", "% retained")
    axes[0].legend(fontsize=8, frameon=False, loc="center right")

    top = np.argsort(-df_counts)[:15]
    axes[1].barh([vocab[i] for i in top][::-1],
                 [100 * df_counts[i] / n for i in top][::-1], color=TEAL, alpha=0.85)
    axes[1].axvline(70, color="#c53030", linewidth=1.4, linestyle="--",
                    label="reference max_df = 0.7")
    axes[1].set_xlim(0, 100)
    style(axes[1], "max_df = 0.7 removes nothing on this corpus",
          "% of documents containing the term")
    axes[1].legend(fontsize=8, frameon=False, loc="lower right")
    axes[1].grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    axes[1].grid(axis="y", visible=False)
    axes[1].tick_params(labelsize=8)

    names = [r["representation"] for r in coupling_rows]
    vals = [r["pair_distance_corr_with_length_gap"] for r in coupling_rows]
    colours = [ORANGE if v > 0.5 else BLUE for v in vals]
    axes[2].barh(names[::-1], vals[::-1], color=colours[::-1], alpha=0.9)
    axes[2].axvline(0, color=INK, linewidth=0.9)
    axes[2].set_xlim(-0.35, 1.0)
    for i, v in enumerate(vals[::-1]):
        axes[2].text(v + (0.03 if v >= 0 else -0.03), i, f"{v:+.2f}", va="center",
                     ha="left" if v >= 0 else "right", fontsize=8, color=INK)
    style(axes[2], "Raw counts encode length. TF IDF does not.",
          "correlation of pairwise distance with length gap")
    axes[2].grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    axes[2].grid(axis="y", visible=False)
    axes[2].tick_params(labelsize=8)

    fig.suptitle("Vectorisation: gates chosen from the data, and the length confound measured",
                 fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "07_vectorisation.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nfigure -> {FIGS / '07_vectorisation.png'}")
    print(f"matrices -> {FEATURES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
