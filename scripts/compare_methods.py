#!/usr/bin/env python3
"""Phase 8. KMeans against NMF and LDA, fitted and scored identically.

Produces:
  reports/method_comparison.csv   every method, K and split, same metrics as phase 7
  reports/topics_k8.csv           what each method's topics actually contain
  reports/figures/13_method_comparison.png

Usage:
    python scripts/compare_methods.py
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cluster import balance, fit_kmeans, length_variance_explained  # noqa: E402
from src.evaluate import score  # noqa: E402
from src.topics import (assignment_confidence, distinctive_topic_terms,  # noqa: E402
                        fit_lda, fit_nmf, hard_assign)

warnings.filterwarnings("ignore", category=UserWarning)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

KS = [2, 8, 11]
INK, GRID = "#1b1b1b", "#d9d9d9"
COLOURS = {"KMeans (tfidf)": "#2b6cb0", "KMeans (counts)": "#dd6b20",
           "NMF (tfidf)": "#2c7a7b", "LDA (counts)": "#6b46c1"}


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=10.5, color=INK, pad=8, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def main() -> int:
    train = pd.read_parquet(FEATURES / "train.parquet")
    test = pd.read_parquet(FEATURES / "test.parquet")
    vocab = json.loads((FEATURES / "tfidf_unigram_vocab.json").read_text())
    lengths = train["n_tokens"].to_numpy()
    y_train = train["product"].to_numpy()
    y_test = test["product"].to_numpy()

    Xt_tr = sparse.load_npz(FEATURES / "tfidf_unigram_train.npz")
    Xt_te = sparse.load_npz(FEATURES / "tfidf_unigram_test.npz")
    Xc_tr = sparse.load_npz(FEATURES / "count_unigram_train.npz")
    Xc_te = sparse.load_npz(FEATURES / "count_unigram_test.npz")
    print(f"train {len(train):,}  test {len(test):,}  vocabulary {len(vocab):,}")

    rows, topic_rows = [], []
    for k in KS:
        print(f"\n--- K = {k} ---")
        fits = {}

        t = time.time()
        km_t = fit_kmeans(Xt_tr, k)
        fits["KMeans (tfidf)"] = (km_t.labels_, km_t.predict(Xt_te),
                                  km_t.cluster_centers_, None, time.time() - t)

        t = time.time()
        km_c = fit_kmeans(Xc_tr, k)
        fits["KMeans (counts)"] = (km_c.labels_, km_c.predict(Xc_te),
                                   km_c.cluster_centers_, None, time.time() - t)

        t = time.time()
        nmf = fit_nmf(Xt_tr, k)
        W_tr, W_te = nmf.transform(Xt_tr), nmf.transform(Xt_te)
        fits["NMF (tfidf)"] = (hard_assign(W_tr), hard_assign(W_te),
                               nmf.components_, W_tr, time.time() - t)

        t = time.time()
        lda = fit_lda(Xc_tr, k)
        D_tr, D_te = lda.transform(Xc_tr), lda.transform(Xc_te)
        fits["LDA (counts)"] = (hard_assign(D_tr), hard_assign(D_te),
                                lda.components_, D_tr, time.time() - t)

        for name, (lab_tr, lab_te, comps, soft, secs) in fits.items():
            for split, labels, truth in [("train", lab_tr, y_train), ("test", lab_te, y_test)]:
                s = score(labels, truth)
                s.update({"method": name, "k": k, "split": split,
                          "fit_seconds": round(secs, 2)})
                if split == "train":
                    s["length_variance_explained"] = round(
                        length_variance_explained(labels, lengths), 4)
                    s.update(balance(labels))
                    if soft is not None:
                        s.update(assignment_confidence(soft))
                rows.append(s)

            used = len(np.unique(lab_tr))
            print(f"  {name:<18} {secs:>5.1f}s  clusters used {used}/{k}  "
                  f"test purity {score(lab_te, y_test)['purity']:.3f}")

            if k == 8:
                terms = distinctive_topic_terms(comps, vocab)
                counts = np.bincount(lab_tr, minlength=k)
                for tpc in range(k):
                    topic_rows.append({
                        "method": name, "topic": tpc,
                        "documents": int(counts[tpc]),
                        "share_pct": round(100 * counts[tpc] / len(train), 1),
                        "distinctive_terms": ", ".join(terms[tpc]),
                    })

    comp = pd.DataFrame(rows)
    comp.to_csv(REPORTS / "method_comparison.csv", index=False)
    pd.DataFrame(topic_rows).to_csv(REPORTS / "topics_k8.csv", index=False)

    test_rows = comp[comp.split == "test"]
    print("\n=== held out results against product labels ===")
    print(f"{'method':<18}{'K':>4}{'purity':>9}{'permuted':>10}{'ARI':>8}{'NMI':>8}")
    for _, r in test_rows.sort_values(["k", "ari"], ascending=[True, False]).iterrows():
        print(f"{r.method:<18}{r.k:>4}{r.purity:>9.3f}{r.purity_permuted:>10.3f}"
              f"{r.ari:>8.3f}{r.nmi:>8.3f}")

    tr = comp[comp.split == "train"]
    print("\n=== does LDA on counts suffer the KMeans length collapse? ===")
    piv = tr.pivot_table(index="k", columns="method", values="length_variance_explained")
    print(piv.round(3).to_string())

    print("\n=== cluster balance and degenerate topics (train) ===")
    print(tr[tr.k == 8][["method", "largest_cluster_pct", "smallest_cluster_pct",
                         "size_ratio", "normalised_entropy"]].to_string(index=False))

    print("\n=== how decisive are the soft assignments? ===")
    soft = tr[tr.mean_top_weight.notna()]
    print(soft[["method", "k", "mean_top_weight", "uniform_weight",
                "share_above_half"]].to_string(index=False))

    # ---------------- figure ----------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    methods = list(COLOURS)
    width = 0.2
    for ax, metric, title in [
        (axes[0], "purity", "Purity against product labels (test)"),
        (axes[1], "ari", "Adjusted Rand index (test)"),
        (axes[2], "nmi", "Normalised mutual information (test)")]:
        for i, m in enumerate(methods):
            t = test_rows[test_rows.method == m].set_index("k").reindex(KS)
            ax.bar(np.arange(len(KS)) + (i - 1.5) * width, t[metric], width=width,
                   color=COLOURS[m], label=m, alpha=0.9)
        ax.set_xticks(range(len(KS)))
        ax.set_xticklabels([f"K={k}" for k in KS])
        style(ax, title)
        if metric == "purity":
            maj = test_rows.purity_majority.iloc[0]
            ax.axhline(maj, color="#c53030", linestyle="--", linewidth=1.2,
                       label=f"majority baseline {maj:.2f}")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("KMeans against NMF and LDA. The reference argues for KMeans without "
                 "fitting either alternative.",
                 fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout()
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / "13_method_comparison.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    print(f"\nreports -> method_comparison.csv, topics_k8.csv")
    print("figure -> 13_method_comparison.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
