#!/usr/bin/env python3
"""Phase 3. Exploratory figures for both corpora, raw and cleaned.

Writes PNGs to reports/figures. Every figure answers a question the audit raised.

Usage:
    python scripts/make_eda_figures.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.dedupe import REDACTION  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
FIGS = ROOT / "reports" / "figures"

INK = "#1b1b1b"
GRID = "#d9d9d9"
BLUE = "#2b6cb0"
ORANGE = "#dd6b20"
TEAL = "#2c7a7b"
RED = "#c53030"
PALETTE = [BLUE, ORANGE, TEAL, RED, "#6b46c1", "#718096",
           "#975a16", "#2f855a", "#b83280", "#4a5568", "#9b2c2c"]


def style(ax, title: str, xlabel: str = "", ylabel: str = "") -> None:
    ax.set_title(title, fontsize=11, color=INK, pad=10, loc="left", fontweight="bold")
    ax.set_xlabel(xlabel, fontsize=9, color=INK)
    ax.set_ylabel(ylabel, fontsize=9, color=INK)
    ax.tick_params(labelsize=8, colors=INK)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.6, alpha=0.7)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / name, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  {name}")


def short(p: str) -> str:
    return {"Credit reporting or other personal consumer reports": "Credit reporting",
            "Money transfer, virtual currency, or money service": "Money transfer",
            "Payday loan, title loan, personal loan, or advance loan": "Payday / personal loan",
            "Checking or savings account": "Checking / savings",
            "Debt or credit management": "Debt / credit mgmt",
            "Vehicle loan or lease": "Vehicle loan"}.get(p, p)


def load():
    d = {}
    for name in ["stratified", "natural"]:
        d[name] = (pd.read_parquet(RAW / f"complaints_{name}.parquet"),
                   pd.read_parquet(PROCESSED / f"corpus_{name}.parquet"))
    return d


# --------------------------------------------------------------------------- figures

def fig_class_distribution(d) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    for ax, name in zip(axes, ["stratified", "natural"]):
        raw, cln = d[name]
        r = raw["product"].value_counts(normalize=True).mul(100)
        c = cln["product"].value_counts(normalize=True).mul(100).reindex(r.index).fillna(0)
        y = np.arange(len(r))
        ax.barh(y + 0.19, r.values, height=0.36, color=BLUE, label="raw")
        ax.barh(y - 0.19, c.values, height=0.36, color=ORANGE, label="after deduplication")
        ax.set_yticks(y)
        ax.set_yticklabels([short(p) for p in r.index], fontsize=8)
        ax.invert_yaxis()
        style(ax, f"{name.capitalize()}  ({len(raw):,} to {len(cln):,} documents)",
              "share of corpus (%)")
        ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
        ax.grid(axis="y", visible=False)
        ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.suptitle("Deduplication reshapes the class balance, and the baseline with it",
                 fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, -0.02,
             "Credit reporting falls from 72.9% to 46.7% of the natural sample once templated "
             "bulk filings are collapsed. The majority class\nbaseline any clustering result "
             "must beat falls with it.", fontsize=9, color="#4a5568", ha="left")
    fig.tight_layout()
    save(fig, "01_class_distribution.png")


def fig_length(d) -> None:
    raw, _ = d["stratified"]
    words = raw["complaint_what_happened"].fillna("").str.split().str.len()
    order = (raw.assign(w=words).groupby("product")["w"].median().sort_values(ascending=False))
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), gridspec_kw={"width_ratios": [1.35, 1]})

    data = [words[raw["product"] == p].values for p in order.index]
    bp = axes[0].boxplot(data, orientation="horizontal", showfliers=False, patch_artist=True, widths=0.6)
    for patch, colour in zip(bp["boxes"], PALETTE):
        patch.set_facecolor(colour); patch.set_alpha(0.75); patch.set_edgecolor(colour)
    for med in bp["medians"]:
        med.set_color(INK); med.set_linewidth(1.4)
    axes[0].set_yticklabels([short(p) for p in order.index], fontsize=8)
    axes[0].invert_yaxis()
    style(axes[0], "Narrative length varies systematically by product", "words per narrative")
    axes[0].grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    axes[0].grid(axis="y", visible=False)

    axes[1].hist(words.clip(upper=900), bins=60, color=BLUE, alpha=0.85)
    axes[1].axvline(words.median(), color=RED, linewidth=1.4,
                    label=f"median {int(words.median())} words")
    style(axes[1], "Overall length distribution", "words per narrative (clipped at 900)",
          "narratives")
    axes[1].legend(fontsize=8, frameon=False)
    fig.text(0.005, -0.02,
             "Mortgage complaints run more than twice the length of debt collection complaints. "
             "Raw counts scale with length, so KMeans\nunder Euclidean distance can separate "
             "these on length alone. This is the case for testing TF IDF against raw counts.",
             fontsize=9, color="#4a5568", ha="left")
    fig.tight_layout()
    save(fig, "02_narrative_length.png")


def fig_templates(d) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for ax, name, colour in zip(axes, ["stratified", "natural"], [BLUE, ORANGE]):
        _, cln = d[name]
        sizes = cln["template_size"].value_counts().sort_index()
        ax.scatter(sizes.index, sizes.values, s=22, color=colour, alpha=0.8)
        ax.set_xscale("log"); ax.set_yscale("log")
        biggest = int(cln["template_size"].max())
        style(ax, f"{name.capitalize()}  (largest template: {biggest} filings)",
              "template size (1 means a unique document)", "number of groups")
        ax.grid(True, which="both", color=GRID, linewidth=0.5, alpha=0.6)
    fig.suptitle("Templated bulk filings follow a heavy tailed distribution",
                 fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.text(0.005, -0.03,
             "Most groups are small, but a long tail of large ones dominates the token counts. "
             "Left unhandled these drive the cluster\ncentroids toward boilerplate phrasing "
             "rather than complaint themes.", fontsize=9, color="#4a5568", ha="left")
    fig.tight_layout()
    save(fig, "03_template_groups.png")


def fig_vocabulary(d) -> None:
    raw, cln = d["stratified"]
    before = Counter()
    for t in raw["complaint_what_happened"].fillna("").str.lower():
        before.update(w for w in t.split() if w.isalpha())
    after = Counter()
    for t in cln["clean_text"]:
        after.update(t.split())

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    for label, counter, colour in [("before cleaning", before, BLUE),
                                   ("after cleaning", after, ORANGE)]:
        freqs = np.array(sorted(counter.values(), reverse=True))
        axes[0].loglog(np.arange(1, len(freqs) + 1), freqs, color=colour, linewidth=1.6,
                       label=f"{label}  ({len(counter):,} types)")
    style(axes[0], "Vocabulary follows Zipf's law, cleaning shifts the curve down",
          "token rank", "frequency")
    axes[0].legend(fontsize=8, frameon=False)
    axes[0].grid(True, which="both", color=GRID, linewidth=0.5, alpha=0.6)

    top = after.most_common(20)[::-1]
    axes[1].barh([t for t, _ in top], [c for _, c in top], color=TEAL, alpha=0.85)
    style(axes[1], "Top 20 tokens after cleaning", "occurrences")
    axes[1].tick_params(labelsize=8)
    axes[1].grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    axes[1].grid(axis="y", visible=False)
    tok_b, tok_a = sum(before.values()), sum(after.values())
    fig.text(0.005, -0.03,
             f"Cleaning removes {100 * (1 - tok_a / tok_b):.0f}% of all tokens but only "
             f"{100 * (1 - len(after) / len(before)):.0f}% of distinct types "
             f"({len(before):,} to {len(after):,}). Stopwords are few in number and\nenormous "
             "in frequency, so the curve drops without the tail shortening. The surviving top "
             "tokens are domain terms, not padding.",
             fontsize=9, color="#4a5568", ha="left")
    fig.tight_layout()
    save(fig, "04_vocabulary.png")


def fig_time_and_redaction(d) -> None:
    raw, _ = d["natural"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    ts = raw.assign(m=raw["date_received"].dt.tz_localize(None).dt.to_period("M").dt.to_timestamp())
    top5 = raw["product"].value_counts().head(5).index
    for p, colour in zip(top5, PALETTE):
        series = ts[ts["product"] == p].groupby("m").size()
        axes[0].plot(series.index, series.values, color=colour, linewidth=1.7, label=short(p))
    style(axes[0], "Monthly volume in the natural sample", "", "complaints sampled")
    axes[0].legend(fontsize=7.5, frameon=False)
    axes[0].tick_params(axis="x", rotation=45)

    red = raw["complaint_what_happened"].fillna("").str.count(REDACTION)
    axes[1].hist(red.clip(upper=40), bins=41, color=ORANGE, alpha=0.85)
    axes[1].axvline(red[red > 0].median(), color=RED, linewidth=1.4,
                    label=f"median {int(red[red > 0].median())} runs where present")
    style(axes[1], "Redaction runs per narrative", "runs of masked text (clipped at 40)",
          "narratives")
    axes[1].legend(fontsize=8, frameon=False)
    fig.text(0.005, -0.03,
             "Sampling is capped per month, so the left panel shows composition rather than "
             "true volume. 81% of narratives carry at\nleast one redaction run, making up 4.25% "
             "of all characters.", fontsize=9, color="#4a5568", ha="left")
    fig.tight_layout()
    save(fig, "05_time_and_redaction.png")


def fig_companies(d) -> None:
    raw, _ = d["natural"]
    top = raw["company"].value_counts().head(15)[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    ax.barh([c[:44] for c in top.index], top.values, color=BLUE, alpha=0.85)
    style(ax, "Most complained about companies, natural sample", "complaints sampled")
    ax.grid(axis="x", color=GRID, linewidth=0.6, alpha=0.7)
    ax.grid(axis="y", visible=False)
    ax.tick_params(labelsize=8)
    fig.text(0.005, -0.04,
             "The credit bureaus dominate, which is the same concentration that drives the "
             "class imbalance and the templated filings.",
             fontsize=9, color="#4a5568", ha="left")
    fig.tight_layout()
    save(fig, "06_top_companies.png")


def main() -> int:
    if not (PROCESSED / "corpus_stratified.parquet").exists():
        print("Run scripts/build_corpus.py first.", file=sys.stderr)
        return 1
    d = load()
    print("writing figures:")
    fig_class_distribution(d)
    fig_length(d)
    fig_templates(d)
    fig_vocabulary(d)
    fig_time_and_redaction(d)
    fig_companies(d)
    print(f"\n6 figures -> {FIGS}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
