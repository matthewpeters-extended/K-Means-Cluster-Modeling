#!/usr/bin/env python3
"""Phase 4. Build the modelling corpus from the raw pulls.

Applies the cleaning pipeline in src/preprocess.py to both corpora and writes
data/processed/corpus_{stratified,natural}.parquet plus a reduction report.

Usage:
    python scripts/build_corpus.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.preprocess import build_corpus  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"
REPORTS = ROOT / "reports"


def main() -> int:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(exist_ok=True)
    all_stats = {}

    for name in ["stratified", "natural"]:
        src = RAW / f"complaints_{name}.parquet"
        if not src.exists():
            print(f"missing {src}. Run scripts/fetch_data.py first.", file=sys.stderr)
            return 1

        df = pd.read_parquet(src)
        started = time.time()
        out, stats = build_corpus(df)
        stats["elapsed_seconds"] = round(time.time() - started, 1)
        stats["retained_pct"] = round(100 * stats["after_near_dedupe"] / stats["input"], 1)
        stats["tokens_median"] = int(out["n_tokens"].median())
        stats["vocabulary"] = len(set(w for t in out["clean_text"] for w in t.split()))

        dest = PROCESSED / f"corpus_{name}.parquet"
        out.to_parquet(dest, index=False)
        all_stats[name] = stats

        print(f"\n=== {name} ===")
        for k, v in stats.items():
            print(f"  {k:<22} {v}")
        print(f"  written                {dest}")

    (REPORTS / "preprocessing_report.json").write_text(json.dumps(all_stats, indent=2))
    pd.DataFrame(all_stats).to_csv(REPORTS / "preprocessing_report.csv")
    print(f"\nreport -> {REPORTS / 'preprocessing_report.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
