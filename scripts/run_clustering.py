#!/usr/bin/env python3
"""Phase 6. Sweep K from 2 to 20 across four representations, then deliver K = 2 and K = 8.

Produces:
  reports/kmeans_sweep.csv          every K, every representation, four internal metrics
  reports/clusters_k2.csv           cluster interpretation at K = 2
  reports/clusters_k8.csv           cluster interpretation at K = 8
  reports/figures/08_k_sweep.png
  reports/figures/09_cluster_sizes.png
  data/processed/features/labels_*.npz   fitted labels, so phase 7 need not refit

Usage:
    python scripts/run_clustering.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import sparse  # noqa: E402
from sklearn.decomposition import TruncatedSVD  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cluster import (RANDOM_STATE, balance, distinctive_terms,  # noqa: E402
                         fit_kmeans, internal_metrics, length_variance_explained,
                         make_dense_sample, top_terms)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

K_RANGE = list(range(2, 21))
HEADLINE_KS = [2, 8]          # the two the brief asks for
SVD_COMPONENTS = 100

INK, GRID = "#1b1b1b", "#d9d9d9"
COLOURS = {"count_unigram": "#dd6b20", "tfidf_unigram": "#2b6cb0",
           "tfidf_bigram": "#2c7a7b", "tfidf_unigram_svd100": "#6b46c1"}


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


def load_representations(train: pd.DataFrame, test: pd.DataFrame) -> dict:
    """Three saved matrices plus a latent semantic variant built here.

    count_unigram_maxdf07 is deliberately excluded: phase 5 V1 proved it is byte identical to
    count_unigram, so clustering it again would only pad the table.
    """
    reps = {}
    for name in ["count_unigram", "tfidf_unigram", "tfidf_bigram"]:
        reps[name] = {
            "train": sparse.load_npz(FEATURES / f"{name}_train.npz"),
            "test": sparse.load_npz(FEATURES / f"{name}_test.npz"),
            "vocab": json.loads((FEATURES / f"{name}_vocab.json").read_text()),
        }

    svd = TruncatedSVD(n_components=SVD_COMPONENTS, random_state=RANDOM_STATE)
    Xtr = svd.fit_transform(reps["tfidf_unigram"]["train"])
    Xte = svd.transform(reps["tfidf_unigram"]["test"])
    reps["tfidf_unigram_svd100"] = {
        "train": Xtr, "test": Xte, "vocab": None, "svd": svd,
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
    }
    print(f"  SVD {SVD_COMPONENTS} components explains "
          f"{100 * reps['tfidf_unigram_svd100']['explained_variance']:.1f}% of variance")
    return reps


def main() -> int:
    if not (FEATURES / "train.parquet").exists():
        print("Run scripts/build_features.py first.", file=sys.stderr)
        return 1
    train = pd.read_parquet(FEATURES / "train.parquet")
    test = pd.read_parquet(FEATURES / "test.parquet")
    lengths = train["n_tokens"].to_numpy()
    print(f"train {len(train):,}   test {len(test):,}")

    reps = load_representations(train, test)

    rows, headline = [], {}
    for name, rep in reps.items():
        X = rep["train"]
        dense_idx, dense_X = make_dense_sample(X)
        print(f"\nsweeping {name} ({X.shape[0]:,} x {X.shape[1]:,})")
        started = time.time()
        for k in K_RANGE:
            km = fit_kmeans(X, k)
            row = {"representation": name, "k": k,
                   "inertia": round(float(km.inertia_), 2),
                   "length_variance_explained": round(
                       length_variance_explained(km.labels_, lengths), 4)}
            row.update(internal_metrics(X, km.labels_, dense_idx, dense_X))
            row.update(balance(km.labels_))
            rows.append(row)

            if k in HEADLINE_KS:
                headline[(name, k)] = km
                np.savez_compressed(
                    FEATURES / f"labels_{name}_k{k}.npz",
                    train=km.labels_, test=km.predict(rep["test"]))
        print(f"  {len(K_RANGE)} fits in {time.time() - started:.1f}s")

    sweep = pd.DataFrame(rows)
    sweep.to_csv(REPORTS / "kmeans_sweep.csv", index=False)

    # ---------------- what the metrics actually recommend ----------------
    print("\n--- best K by each internal metric ---")
    print(f"{'representation':<24}{'silhouette':>12}{'Davies Bouldin':>16}{'Calinski H':>13}")
    for name in reps:
        s = sweep[sweep.representation == name]
        print(f"{name:<24}{int(s.loc[s.silhouette.idxmax(), 'k']):>12}"
              f"{int(s.loc[s.davies_bouldin.idxmin(), 'k']):>16}"
              f"{int(s.loc[s.calinski_harabasz.idxmax(), 'k']):>13}")

    print("\n--- the V2 falsification test: length variance explained by cluster ---")
    piv = sweep.pivot_table(index="k", columns="representation",
                            values="length_variance_explained")
    print(piv.loc[[2, 8, 20]].round(3).to_string())

    # ---------------- interpret K = 2 and K = 8 ----------------
    for k in HEADLINE_KS:
        out = []
        for name, rep in reps.items():
            km = headline[(name, k)]
            counts = np.bincount(km.labels_, minlength=k)
            vocab = rep["vocab"]
            if vocab is None:
                # map SVD centroids back into term space so clusters can still be named
                terms_top = terms_dist = None
                back = rep["svd"].inverse_transform(km.cluster_centers_)
                v = reps["tfidf_unigram"]["vocab"]
                order = back.argsort()[:, ::-1]
                contrast = (back - back.mean(axis=0, keepdims=True)).argsort()[:, ::-1]
                terms_top = {c: [v[i] for i in order[c, :12]] for c in range(k)}
                terms_dist = {c: [v[i] for i in contrast[c, :12]] for c in range(k)}
            else:
                terms_top = top_terms(km, vocab)
                terms_dist = distinctive_terms(km, vocab)
            for c in range(k):
                out.append({
                    "representation": name, "k": k, "cluster": c,
                    "documents": int(counts[c]),
                    "share_pct": round(100 * counts[c] / len(train), 1),
                    "top_terms": ", ".join(terms_top[c]),
                    "distinctive_terms": ", ".join(terms_dist[c]),
                })
        pd.DataFrame(out).to_csv(REPORTS / f"clusters_k{k}.csv", index=False)

    print("\n--- K = 8 on tfidf_unigram, distinctive terms per cluster ---")
    k8 = pd.read_csv(REPORTS / "clusters_k8.csv")
    for _, r in k8[k8.representation == "tfidf_unigram"].iterrows():
        print(f"  cluster {r.cluster}  {r.share_pct:>5.1f}%  {r.distinctive_terms[:88]}")

    # ---------------- figures ----------------
    fig, axes = plt.subplots(2, 2, figsize=(13, 8))
    panels = [("inertia", "Elbow: within cluster sum of squares", False),
              ("silhouette", "Silhouette (higher is better)", False),
              ("davies_bouldin", "Davies Bouldin (lower is better)", False),
              ("calinski_harabasz", "Calinski Harabasz (higher is better)", False)]
    for ax, (col, title, _) in zip(axes.ravel(), panels):
        for name in reps:
            s = sweep[sweep.representation == name]
            y = s[col] / s[col].iloc[0] if col == "inertia" else s[col]
            ax.plot(s.k, y, marker="o", markersize=3.5, linewidth=1.5,
                    color=COLOURS[name], label=name)
        for kk in HEADLINE_KS:
            ax.axvline(kk, color="#c53030", linewidth=0.9, linestyle="--", alpha=0.7)
        style(ax, title, "K", "relative to K=2" if col == "inertia" else "")
        ax.set_xticks(range(2, 21, 2))
    axes[0][0].legend(fontsize=7.5, frameon=False)
    fig.suptitle("K sweep, 2 to 20. Dashed lines mark the two values the brief asks for.",
                 fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / "08_k_sweep.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax, col, title in [
        (axes[0], "length_variance_explained",
         "Share of document length variance explained by the clustering"),
        (axes[1], "size_ratio", "Cluster size imbalance (largest divided by smallest)")]:
        for name in reps:
            s = sweep[sweep.representation == name]
            ax.plot(s.k, s[col], marker="o", markersize=3.5, linewidth=1.5,
                    color=COLOURS[name], label=name)
        style(ax, title, "K")
        ax.set_xticks(range(2, 21, 2))
    axes[0].axhline(0.1, color="#c53030", linestyle="--", linewidth=1,
                    label="10%: above this the partition is partly about length")
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[1].set_yscale("log")
    fig.suptitle("Phase 5 predicted raw counts would cluster on length. This is the test.",
                 fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / "09_cluster_diagnostics.png", dpi=160, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)

    # ---------------- the length band figure ----------------
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), sharey=False)
    for ax, name, colour in [(axes[0], "count_unigram", COLOURS["count_unigram"]),
                             (axes[1], "tfidf_unigram", COLOURS["tfidf_unigram"])]:
        lab = np.load(FEATURES / f"labels_{name}_k8.npz")["train"]
        t = (pd.DataFrame({"c": lab, "tokens": lengths})
             .groupby("c")["tokens"].agg(["count", "median"]).sort_values("median"))
        bars = ax.bar(range(len(t)), t["median"], color=colour, alpha=0.85)
        for i, (bar, (_, r)) in enumerate(zip(bars, t.iterrows())):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                    f"n={int(r['count']):,}", ha="center", va="bottom", fontsize=7.5, color=INK)
        ax.set_xticks(range(len(t)))
        ax.set_xticklabels([f"c{i}" for i in t.index], fontsize=8)
        style(ax, f"{name}, K = 8", "cluster, ordered by median length",
              "median words per document")
    axes[0].set_ylim(0, 2500)
    axes[1].set_ylim(0, 2500)
    fig.suptitle("Raw counts produce length bands. TF IDF produces topics.",
                 fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, -0.13,
             "Left: cluster medians climb from 45 to 2,238 words, and the largest cluster holds "
             "9,870 documents while the smallest holds 4. That is a\npartition of the corpus by "
             "length wearing the costume of topic discovery. Right: the same algorithm on L2 "
             "normalised TF IDF spans 30 to 157\nwords with cluster sizes between 739 and 3,380.",
             fontsize=9, color="#4a5568", ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / "10_length_bands.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nsweep -> {REPORTS / 'kmeans_sweep.csv'}")
    print("figures -> 08_k_sweep.png, 09_cluster_diagnostics.png, 10_length_bands.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
