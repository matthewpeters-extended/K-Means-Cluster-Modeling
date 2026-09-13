#!/usr/bin/env python3
"""Phase 7. Score every clustering against the bureau's hand assigned labels.

Produces:
  reports/evaluation.csv              every representation, K, split and label scheme
  reports/contingency_k8_tfidf.csv    the table behind the headline number
  reports/company_confound.csv        how much of the clustering is firm recognition
  reports/figures/11_evaluation.png
  reports/figures/12_contingency.png

Usage:
    python scripts/evaluate_clusters.py
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
from sklearn.metrics import normalized_mutual_info_score  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.evaluate import (brand_share, company_tokens, contingency,  # noqa: E402
                          majority_baseline, score)

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "data" / "processed" / "features"
REPORTS = ROOT / "reports"
FIGS = REPORTS / "figures"

REPRESENTATIONS = ["count_unigram", "tfidf_unigram", "tfidf_bigram", "tfidf_unigram_svd100"]
KS = [2, 8, 11]
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


def short(p: str) -> str:
    return {"Credit reporting or other personal consumer reports": "Credit reporting",
            "Money transfer, virtual currency, or money service": "Money transfer",
            "Payday loan, title loan, personal loan, or advance loan": "Payday / personal",
            "Checking or savings account": "Checking / savings",
            "Debt or credit management": "Debt / credit mgmt",
            "Vehicle loan or lease": "Vehicle loan"}.get(p, p)


def main() -> int:
    train = pd.read_parquet(FEATURES / "train.parquet")
    test = pd.read_parquet(FEATURES / "test.parquet")
    print(f"train {len(train):,}   test {len(test):,}")
    print(f"majority baseline  train {majority_baseline(train['product'].to_numpy()):.4f}   "
          f"test {majority_baseline(test['product'].to_numpy()):.4f}")

    rows = []
    for rep in REPRESENTATIONS:
        for k in KS:
            saved = np.load(FEATURES / f"labels_{rep}_k{k}.npz")
            for split, df, labels in [("train", train, saved["train"]),
                                      ("test", test, saved["test"])]:
                for scheme in ["product", "issue"]:
                    s = score(labels, df[scheme].fillna("unknown").to_numpy())
                    s.update({"representation": rep, "k": k, "split": split,
                              "label_scheme": scheme})
                    rows.append(s)

    ev = pd.DataFrame(rows)
    cols = ["representation", "k", "split", "label_scheme", "purity", "purity_permuted",
            "purity_majority", "beats_majority", "ari", "nmi", "homogeneity", "completeness"]
    ev = ev[cols + [c for c in ev.columns if c not in cols]]
    ev.to_csv(REPORTS / "evaluation.csv", index=False)

    prod = ev[(ev.label_scheme == "product")]
    print("\n=== against PRODUCT labels ===")
    for split in ["train", "test"]:
        print(f"\n--- {split} ---")
        t = prod[prod.split == split]
        print(f"{'representation':<22}{'K':>4}{'purity':>9}{'permuted':>10}{'majority':>10}"
              f"{'ARI':>8}{'NMI':>8}")
        for _, r in t.iterrows():
            flag = "" if r.purity > r.purity_majority else "   <- below majority baseline"
            print(f"{r.representation:<22}{r.k:>4}{r.purity:>9.3f}{r.purity_permuted:>10.3f}"
                  f"{r.purity_majority:>10.3f}{r.ari:>8.3f}{r.nmi:>8.3f}{flag}")

    # ---------- does the train result survive on held out data? ----------
    print("\n=== train to test generalisation (product, ARI) ===")
    piv = prod.pivot_table(index=["representation", "k"], columns="split", values="ari")
    piv["drop"] = (piv["train"] - piv["test"]).round(4)
    print(piv.round(4).to_string())

    # ---------- the contingency table behind the headline ----------
    lab = np.load(FEATURES / "labels_tfidf_unigram_k8.npz")["train"]
    ct = contingency(lab, train["product"].to_numpy())
    ct.columns = [short(c) for c in ct.columns]
    ct.to_csv(REPORTS / "contingency_k8_tfidf.csv")
    print("\n=== contingency, tfidf_unigram K=8, train (row percentages) ===")
    pct = (100 * ct.div(ct.sum(axis=1), axis=0)).round(1)
    print(pct.to_string())

    # ---------- open question 1: is this company recognition? ----------
    brands = company_tokens(pd.concat([train["company"], test["company"]]))
    k8 = pd.read_csv(REPORTS / "clusters_k8.csv")
    conf_rows = []
    for rep in REPRESENTATIONS:
        sub = k8[(k8.representation == rep) & (k8.k == 8)]
        distinctive = {int(r.cluster): r.distinctive_terms.split(", ") for _, r in sub.iterrows()}
        bs = brand_share(distinctive, brands)
        labels = np.load(FEATURES / f"labels_{rep}_k8.npz")["train"]
        conf_rows.append({
            "representation": rep,
            "brand_share_of_distinctive_terms": bs["overall"],
            "nmi_with_product": round(normalized_mutual_info_score(
                train["product"].to_numpy(), labels), 4),
            "nmi_with_company": round(normalized_mutual_info_score(
                train["company"].fillna("unknown").to_numpy(), labels), 4),
        })
    conf = pd.DataFrame(conf_rows)
    conf["company_over_product"] = (conf.nmi_with_company / conf.nmi_with_product).round(2)
    conf.to_csv(REPORTS / "company_confound.csv", index=False)
    print("\n=== open question 1: company recognition vs complaint semantics (K=8, train) ===")
    print(conf.to_string(index=False))

    # ---------- open question 2: the register split ----------
    print("\n=== open question 2: clusters 3 and 4 of tfidf_unigram K=8 ===")
    for c in [3, 4]:
        sel = train[lab == c]
        top = sel["product"].value_counts(normalize=True).head(3)
        print(f"  cluster {c} ({len(sel):,} docs): " +
              ", ".join(f"{short(p)} {100*v:.0f}%" for p, v in top.items()))
    merged = lab.copy()
    merged[merged == 4] = 3
    from src.evaluate import purity as _purity
    truth = train["product"].to_numpy()
    print(f"  purity with 3 and 4 separate : {_purity(lab, truth):.4f}")
    print(f"  purity with 3 and 4 merged   : {_purity(merged, truth):.4f}")

    # ---------- figures ----------
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6))
    test_prod = prod[prod.split == "test"]
    for ax, metric, title in [
        (axes[0], "purity", "Purity against product labels (test)"),
        (axes[1], "ari", "Adjusted Rand index (test)"),
        (axes[2], "nmi", "Normalised mutual information (test)")]:
        width = 0.2
        for i, rep in enumerate(REPRESENTATIONS):
            t = test_prod[test_prod.representation == rep]
            ax.bar(np.arange(len(KS)) + (i - 1.5) * width, t[metric], width=width,
                   color=COLOURS[rep], label=rep, alpha=0.9)
        ax.set_xticks(range(len(KS)))
        ax.set_xticklabels([f"K={k}" for k in KS])
        style(ax, title)
        if metric == "purity":
            maj = test_prod.purity_majority.iloc[0]
            ax.axhline(maj, color="#c53030", linestyle="--", linewidth=1.2,
                       label=f"majority baseline {maj:.2f}")
            perm = test_prod.groupby("k").purity_permuted.mean()
            ax.plot(range(len(KS)), perm.values, marker="_", markersize=22, linestyle="",
                    color=INK, label="permuted control")
        ax.legend(fontsize=7, frameon=False)
    fig.suptitle("Ground truth evaluation. Every number sits next to the baseline it must beat.",
                 fontsize=13, fontweight="bold", color=INK, x=0.005, ha="left")
    fig.tight_layout()
    fig.savefig(FIGS / "11_evaluation.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5.2))
    im = ax.imshow(pct.values, cmap="Blues", aspect="auto", vmin=0, vmax=100)
    ax.set_xticks(range(len(pct.columns)))
    ax.set_xticklabels(pct.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(pct.index)))
    ax.set_yticklabels([f"cluster {i}" for i in pct.index], fontsize=8)
    for i in range(pct.shape[0]):
        for j in range(pct.shape[1]):
            v = pct.values[i, j]
            if v >= 5:
                ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5,
                        color="white" if v > 50 else INK)
    ax.set_title("Where each cluster's documents actually came from, tfidf_unigram K = 8 "
                 "(row percentages)", fontsize=11, fontweight="bold", color=INK, loc="left",
                 pad=10)
    fig.colorbar(im, ax=ax, label="% of cluster")
    fig.tight_layout()
    fig.savefig(FIGS / "12_contingency.png", dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"\nreports -> evaluation.csv, contingency_k8_tfidf.csv, company_confound.csv")
    print("figures -> 11_evaluation.png, 12_contingency.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
