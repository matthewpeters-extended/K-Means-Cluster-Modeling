#!/usr/bin/env python3
"""Fetch consumer finance complaint narratives from the CFPB public API.

Builds two corpora so the effect of class imbalance can be measured rather than assumed:

  stratified : capped per product per month, so small products are not drowned out
  natural    : proportional to how complaints actually arrive, credit reporting and all

Neither file is committed. This script plus fetch_metadata.json is the reproducible record.

Usage:
    python scripts/fetch_data.py --dry-run
    python scripts/fetch_data.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from datetime import date
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

API = "https://www.consumerfinance.gov/data-research/consumer-complaints/search/api/v1/"

# Verified live against the API on 2026-09-12. Passing format=json returns 404; JSON is default.
PRODUCTS = [
    "Credit reporting or other personal consumer reports",
    "Debt collection",
    "Credit card",
    "Checking or savings account",
    "Mortgage",
    "Money transfer, virtual currency, or money service",
    "Student loan",
    "Vehicle loan or lease",
    "Payday loan, title loan, personal loan, or advance loan",
    "Prepaid card",
    "Debt or credit management",
]

KEEP_FIELDS = [
    "complaint_id",
    "date_received",
    "product",
    "sub_product",
    "issue",
    "sub_issue",
    "complaint_what_happened",
    "company",
    "state",
    "submitted_via",
    "company_response",
    "timely",
]

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def month_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Inclusive month boundaries between two YYYY-MM strings."""
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        first = date(y, m, 1)
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        last = date(ny, nm, 1) - pd.Timedelta(days=1).to_pytimedelta()
        out.append((first.isoformat(), last.isoformat()))
        y, m = ny, nm
    return out


def query(session: requests.Session, size: int, lo: str, hi: str,
          product: str | None = None, retries: int = 4) -> list[dict]:
    params = {
        "size": size,
        "no_aggs": "true",
        "has_narrative": "true",
        "date_received_min": lo,
        "date_received_max": hi,
    }
    if product:
        params["product"] = product

    for attempt in range(retries):
        try:
            r = session.get(API, params=params, timeout=120)
            # Client errors other than rate limiting will not fix themselves, so fail fast.
            if 400 <= r.status_code < 500 and r.status_code != 429:
                r.raise_for_status()
            r.raise_for_status()
            return [h["_source"] for h in r.json()["hits"]["hits"]]
        except requests.HTTPError as exc:
            code = exc.response.status_code if exc.response is not None else 0
            if 400 <= code < 500 and code != 429:
                raise
            if attempt == retries - 1:
                raise
            _backoff(attempt, exc)
        except Exception as exc:  # noqa: BLE001 - transient network, retry then give up loudly
            if attempt == retries - 1:
                raise
            _backoff(attempt, exc)
    return []


def _backoff(attempt: int, exc: Exception) -> None:
    wait = 2 ** attempt
    print(f"  retry {attempt + 1} after {exc.__class__.__name__}, sleeping {wait}s",
          file=sys.stderr)
    time.sleep(wait)


def tidy(rows: list[dict]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    for col in KEEP_FIELDS:
        if col not in df.columns:
            df[col] = None
    df = df[KEEP_FIELDS]
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce", utc=True)
    df = df.dropna(subset=["complaint_what_happened"])
    df = df[df["complaint_what_happened"].str.strip().str.len() > 0]
    return df.drop_duplicates(subset="complaint_id").reset_index(drop=True)


def fetch_stratified(session, windows, per_stratum) -> pd.DataFrame:
    rows = []
    bar = tqdm(total=len(windows) * len(PRODUCTS), desc="stratified", unit="stratum")
    for lo, hi in windows:
        for product in PRODUCTS:
            rows.extend(query(session, per_stratum, lo, hi, product))
            bar.update(1)
    bar.close()
    return tidy(rows)


def fetch_natural(session, windows, per_month) -> pd.DataFrame:
    rows = []
    for lo, hi in tqdm(windows, desc="natural", unit="month"):
        rows.extend(query(session, per_month, lo, hi))
    return tidy(rows)


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-01", help="first month, YYYY-MM")
    ap.add_argument("--end", default="2024-12", help="last month, YYYY-MM")
    ap.add_argument("--per-stratum", type=int, default=75,
                    help="max narratives per product per month")
    ap.add_argument("--per-month", type=int, default=825,
                    help="narratives per month for the natural distribution sample")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without calling the API")
    args = ap.parse_args()

    windows = month_windows(args.start, args.end)
    strata = len(windows) * len(PRODUCTS)

    print(f"months          : {len(windows)}  ({args.start} to {args.end})")
    print(f"products        : {len(PRODUCTS)}")
    print(f"strata          : {strata}")
    print(f"stratified cap  : {args.per_stratum} per stratum, "
          f"upper bound {strata * args.per_stratum:,} rows")
    print(f"natural sample  : {args.per_month} per month, "
          f"upper bound {len(windows) * args.per_month:,} rows")
    print(f"requests        : {strata + len(windows)}")

    if args.dry_run:
        print("\ndry run, nothing fetched")
        return 0

    RAW.mkdir(parents=True, exist_ok=True)
    # Do not set a custom User-Agent here. The consumerfinance.gov edge returns 403 for both
    # custom agent strings and browser-like ones; the stock requests agent is accepted.
    session = requests.Session()

    started = time.time()
    strat = fetch_stratified(session, windows, args.per_stratum)
    nat = fetch_natural(session, windows, args.per_month)

    strat_path = RAW / "complaints_stratified.parquet"
    nat_path = RAW / "complaints_natural.parquet"
    strat.to_parquet(strat_path, index=False)
    nat.to_parquet(nat_path, index=False)

    meta = {
        "fetched_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "elapsed_seconds": round(time.time() - started, 1),
        "api": API,
        "filters": {"has_narrative": True, "start": args.start, "end": args.end},
        "products": PRODUCTS,
        "per_stratum": args.per_stratum,
        "per_month": args.per_month,
        "files": {
            "stratified": {
                "path": strat_path.name,
                "rows": int(len(strat)),
                "sha256": checksum(strat_path),
                "product_counts": strat["product"].value_counts().to_dict(),
            },
            "natural": {
                "path": nat_path.name,
                "rows": int(len(nat)),
                "sha256": checksum(nat_path),
                "product_counts": nat["product"].value_counts().to_dict(),
            },
        },
    }
    (RAW / "fetch_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nstratified : {len(strat):,} rows -> {strat_path}")
    print(f"natural    : {len(nat):,} rows -> {nat_path}")
    print(f"metadata   : {RAW / 'fetch_metadata.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
