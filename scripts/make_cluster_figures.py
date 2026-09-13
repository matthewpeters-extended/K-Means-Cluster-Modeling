#!/usr/bin/env python3
"""Phase 9. The visual deliverables the brief asks for, plus the quantitative complements.

Word clouds are what the brief and the reference solution ask for, and they are produced here.
They are also a weak evidence format: they show which terms are frequent and hide the ranking,
the magnitude and the contrast between clusters. So every word cloud is paired with a ranked
term chart showing the same information in a form you can actually read numbers off.

Produces:
  reports/figures/14_wordclouds_k2.png
  reports/figures/15_wordclouds_k8.png
  reports/figures/16_top_terms_k8.png
  reports/figures/17_tsne.png
  reports/figures/18_sentiment.png
  reports/sentiment_by_cluster.csv

Usage:
    python scripts/make_cluster_figures.py
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
from sklearn.manifold import TSNE  # noqa: E402
from wordcloud import WordCloud  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cluster import distinctive_terms, fit_kmeans  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

INK, GRID = "#1b1b1b", "#d9d9d9"
PALETTE = ["#2b6cb0", "#dd6b20", "#2c7a7b", "#c53030", "#6b46c1",
           "#975a16", "#2f855a", "#b83280", "#4a5568", "#9b2c2c", "#285e61"]
TSNE_SAMPLE = 5000

# Names read off the distinctive terms in phase 6 C5, kept here so figures are legible.
CLUSTER_NAMES_K8 = {
    0: "Debt collection practices",
    1: "Loan and mortgage servicing",
    2: "Vehicle finance and dealers",
    3: "Credit report accuracy",
    4: "Statutory dispute template",
    5: "Student loan servicing",
    6: "Banking and transfers",
    7: "Card declines and gift cards",
}
CLUSTER_NAMES_K2 = {0: "Credit reporting and debt", 1: "Money, cards and banking"}


def style(ax, title, xlabel="", ylabel=""):
    ax.set_title(title, fontsize=10, color=INK, pad=8, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.grid(color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def short(p: str) -> str:
    return {"Credit reporting or other personal consumer reports": "Credit reporting",
            "Money transfer, virtual currency, or money service": "Money transfer",
            "Payday loan, title loan, personal loan, or advance loan": "Payday / personal",
            "Checking or savings account": "Checking / savings",
            "Debt or credit management": "Debt / credit mgmt",
            "Vehicle loan or lease": "Vehicle loan"}.get(p, p)


def centroid_weights(model, vocab: list[str], cluster: int, n: int = 120) -> dict[str, float]:
    """Term weights for one centroid, as a frequency map for the word cloud.

    Built from the centroid rather than by concatenating the cluster's raw text. Concatenating
    lets a handful of long documents dominate the cloud, which is the same length confound
    phase 5 measured, reappearing in the visualisation layer.
    """
    w = model.cluster_centers_[cluster]
    top = np.argsort(-w)[:n]
    return {vocab[i]: float(w[i]) for i in top if w[i] > 0}


def draw_wordclouds(model, vocab, names, path: Path, cols: int, title: str, subtitle: str):
    k = model.n_clusters
    rows = int(np.ceil(k / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4.6 * cols, 3.6 * rows))
    axes = np.atleast_1d(axes).ravel()
    counts = np.bincount(model.labels_, minlength=k)

    for c in range(k):
        wc = WordCloud(width=900, height=560, background_color="white",
                       colormap="viridis", prefer_horizontal=0.92,
                       max_words=90, relative_scaling=0.45, random_state=42)
        wc.generate_from_frequencies(centroid_weights(model, vocab, c))
        axes[c].imshow(wc, interpolation="bilinear")
        axes[c].axis("off")
        axes[c].set_title(f"{c}. {names.get(c, f'Cluster {c}')}   "
                          f"{100 * counts[c] / counts.sum():.1f}%  (n={counts[c]:,})",
                          fontsize=10, color=INK, loc="left", pad=6, fontweight="bold")
    for extra in range(k, len(axes)):
        axes[extra].axis("off")

    fig.suptitle(title, fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, -0.05, subtitle, fontsize=9, color="#4a5568", ha="left")
    fig.tight_layout(h_pad=3.0)
    fig.savefig(path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {path.name}")


def main() -> int:
    train = pd.read_parquet(FEATURES / "train.parquet")
    vocab = json.loads((FEATURES / "tfidf_unigram_vocab.json").read_text())
    X = sparse.load_npz(FEATURES / "tfidf_unigram_train.npz")
    FIGS.mkdir(parents=True, exist_ok=True)
    print(f"train {len(train):,}")

    km2 = fit_kmeans(X, 2)
    km8 = fit_kmeans(X, 8)

    print("word clouds:")
    draw_wordclouds(km2, vocab, CLUSTER_NAMES_K2, FIGS / "14_wordclouds_k2.png", 2,
                    "K = 2: the broad split the brief asks for",
                    "Built from TF IDF centroid weights, not by concatenating each cluster's "
                    "text. Concatenation lets the longest documents dominate,\nwhich is the "
                    "length confound of phase 5 reappearing in the visualisation.")
    draw_wordclouds(km8, vocab, CLUSTER_NAMES_K8, FIGS / "15_wordclouds_k8.png", 4,
                    "K = 8: the granular split the brief asks for",
                    "Word size is centroid weight. Compare with figure 16, which shows the same "
                    "information as ranked magnitudes: the cloud hides\nhow far ahead the top "
                    "term is, and hides which terms are shared with every other cluster.")

    # ---------- ranked terms, the quantitative complement ----------
    dist = distinctive_terms(km8, vocab, n=10)
    counts = np.bincount(km8.labels_, minlength=8)
    centres = km8.cluster_centers_
    contrast = centres - centres.mean(axis=0, keepdims=True)
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    for c, ax in enumerate(axes.ravel()):
        terms = dist[c]
        vals = [contrast[c, vocab.index(t)] for t in terms]
        ax.barh(terms[::-1], vals[::-1], color=PALETTE[c], alpha=0.88)
        style(ax, f"{c}. {CLUSTER_NAMES_K8[c]}\n{100 * counts[c] / counts.sum():.1f}% "
                  f"(n={counts[c]:,})", "weight above the average centroid")
        ax.grid(axis="y", visible=False)
        ax.tick_params(labelsize=8)
    fig.suptitle("The same eight clusters as ranked magnitudes rather than clouds",
                 fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, -0.02,
             "Each bar is how far a term's weight in this centroid exceeds its average across "
             "all eight. This is what a word cloud cannot show:\nthe gap between the first and "
             "second term, and whether a term is characteristic of the cluster or merely common "
             "everywhere.", fontsize=9, color="#4a5568", ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / "16_top_terms_k8.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  16_top_terms_k8.png")

    # ---------- TSNE ----------
    print("TSNE:")
    Xs = np.load(FEATURES / "svd_train.npy") if (FEATURES / "svd_train.npy").exists() else None
    if Xs is None:
        from sklearn.decomposition import TruncatedSVD
        Xs = TruncatedSVD(n_components=100, random_state=42).fit_transform(X)
    rng = np.random.default_rng(0)
    idx = rng.choice(len(train), size=min(TSNE_SAMPLE, len(train)), replace=False)
    t0 = time.time()
    emb = TSNE(n_components=2, perplexity=30, init="pca", random_state=42,
               max_iter=1000).fit_transform(Xs[idx])
    print(f"  {len(idx):,} points in {time.time() - t0:.0f}s")

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    lab8 = km8.labels_[idx]
    for c in range(8):
        m = lab8 == c
        axes[0].scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.55, color=PALETTE[c],
                        label=f"{c}. {CLUSTER_NAMES_K8[c]}", linewidths=0)
    axes[0].legend(fontsize=7, frameon=False, markerscale=3, loc="upper right")
    axes[0].set_title("Coloured by KMeans cluster (K = 8)", fontsize=11, fontweight="bold",
                      color=INK, loc="left", pad=8)

    prods = train["product"].to_numpy()[idx]
    for i, p in enumerate(pd.Series(prods).value_counts().index):
        m = prods == p
        axes[1].scatter(emb[m, 0], emb[m, 1], s=4, alpha=0.55, color=PALETTE[i % len(PALETTE)],
                        label=short(p), linewidths=0)
    axes[1].legend(fontsize=7, frameon=False, markerscale=3, loc="upper right")
    axes[1].set_title("Coloured by true product label", fontsize=11, fontweight="bold",
                      color=INK, loc="left", pad=8)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
    fig.suptitle("The same 5,000 documents, coloured two ways",
                 fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, -0.02,
             "Left is tidy by construction: KMeans drew those boundaries. Right is the honest "
             "picture, and the overlap between products is\nwhy held out purity is 0.47 rather "
             "than 0.9. TSNE preserves local neighbourhoods only; distances between distant "
             "blobs mean nothing.", fontsize=9, color="#4a5568", ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / "17_tsne.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  17_tsne.png")

    # ---------- sentiment ----------
    print("sentiment:")
    from nltk.sentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()
    t0 = time.time()
    comp = np.array([sia.polarity_scores(t)["compound"]
                     for t in train["complaint_what_happened"].fillna("")])
    print(f"  {len(comp):,} narratives scored in {time.time() - t0:.0f}s")

    sent = (pd.DataFrame({"cluster": km8.labels_, "compound": comp})
            .groupby("cluster")["compound"]
            .agg(["count", "mean", "median",
                  ("pct_negative", lambda s: float((s < -0.05).mean() * 100)),
                  ("pct_positive", lambda s: float((s > 0.05).mean() * 100))]).round(3))
    sent["theme"] = [CLUSTER_NAMES_K8[c] for c in sent.index]
    sent.to_csv(REPORTS / "sentiment_by_cluster.csv")
    print(sent.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    axes[0].hist(comp, bins=60, color="#2b6cb0", alpha=0.85)
    axes[0].axvline(0, color=INK, linewidth=1)
    axes[0].axvline(comp.mean(), color="#c53030", linewidth=1.4,
                    label=f"mean {comp.mean():+.3f}")
    style(axes[0], "VADER compound score across all complaints", "compound score",
          "narratives")
    axes[0].legend(fontsize=8, frameon=False)

    order = sent.sort_values("mean")
    axes[1].barh([f"{i}. {CLUSTER_NAMES_K8[i]}" for i in order.index], order["mean"],
                 color=["#c53030" if v < 0 else "#2f855a" for v in order["mean"]], alpha=0.88)
    axes[1].axvline(0, color=INK, linewidth=0.9)
    style(axes[1], "Mean sentiment by theme", "mean VADER compound score")
    axes[1].grid(axis="y", visible=False)
    axes[1].tick_params(labelsize=8)
    fig.suptitle("Sentiment intensity per theme, and why this layer should be read sceptically",
                 fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, -0.04,
             f"Every document here is a complaint, yet {100 * (comp > 0.05).mean():.0f}% score "
             f"positive and the corpus mean is {comp.mean():+.3f}. VADER is tuned on social "
             "media, where\nintensity is carried by punctuation, capitals and emoji. Formal "
             "complaint prose carries it through narrative instead, which VADER cannot read.",
             fontsize=9, color="#4a5568", ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / "18_sentiment.png", dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("  18_sentiment.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
