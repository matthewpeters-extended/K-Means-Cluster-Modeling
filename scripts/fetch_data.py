#!/usr/bin/env python3
"""Fetch consumer finance complaint narratives from the CFPB public API.

Builds two corpora so the effect of class imbalance can be measured rather than assumed:

  stratified : capped per product per month, so small products are not drowned out
  natural    : proportional to how complaints actually arrive, credit reporting and all

Every stratum is checkpointed to data/raw/_shards as soon as it lands, so an interrupted
run resumes where it stopped instead of starting over. Reruns are cheap and safe.

Neither parquet file is committed. This script plus fetch_metadata.json is the reproducible
record of how the corpus was built.

Usage:
    python scripts/fetch_data.py --dry-run
    python scripts/fetch_data.py
    python scripts/fetch_data.py --fresh      # ignore checkpoints and refetch everything
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import sys
import time
from datetime import date, timedelta
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
SHARDS = RAW / "_shards"


# --------------------------------------------------------------------------- windows

def month_windows(start: str, end: str) -> list[tuple[str, str]]:
    """Inclusive first and last day for every month between two YYYY-MM strings."""
    sy, sm = (int(x) for x in start.split("-"))
    ey, em = (int(x) for x in end.split("-"))
    out: list[tuple[str, str]] = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        ny, nm = (y + 1, 1) if m == 12 else (y, m + 1)
        out.append((date(y, m, 1).isoformat(), (date(ny, nm, 1) - timedelta(days=1)).isoformat()))
        y, m = ny, nm
    return out


# --------------------------------------------------------------------------- checkpoints

def shard_path(kind: str, label: str | None, lo: str) -> Path:
    slug = re.sub(r"[^a-z0-9]+", "_", (label or "all").lower()).strip("_")[:60]
    return SHARDS / f"{kind}__{slug}__{lo[:7]}.jsonl"


def read_shard(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def write_shard(path: Path, rows: list[dict]) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(json.dumps(r) for r in rows))
    tmp.replace(path)  # atomic, so a kill mid-write cannot leave a half shard


# --------------------------------------------------------------------------- fetching

class RateLimited(Exception):
    pass


def query(session: requests.Session, size: int, lo: str, hi: str,
          product: str | None, sleep: float, retries: int = 6) -> list[dict]:
    """One API call, with backoff that treats rate limiting as expected rather than fatal."""
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

            if r.status_code == 429:
                # Honour the server's own guidance when it gives any, otherwise back off hard.
                hinted = r.headers.get("Retry-After")
                wait = float(hinted) if hinted and hinted.isdigit() else min(120, 5 * 2 ** attempt)
                raise RateLimited(f"rate limited, waiting {wait:.0f}s")

            # Other client errors will not fix themselves, so fail loudly and immediately.
            r.raise_for_status()
            time.sleep(sleep)
            return [h["_source"] for h in r.json()["hits"]["hits"]]

        except RateLimited as exc:
            if attempt == retries - 1:
                raise
            hinted = None
            m = re.search(r"waiting (\d+)s", str(exc))
            wait = int(m.group(1)) if m else 30
            tqdm.write(f"  {exc}  (attempt {attempt + 1} of {retries})")
            time.sleep(wait)

        except requests.HTTPError:
            raise

        except Exception as exc:  # noqa: BLE001 - transient network, retry then give up loudly
            if attempt == retries - 1:
                raise
            wait = min(60, 2 ** attempt)
            tqdm.write(f"  retry {attempt + 1} after {exc.__class__.__name__}, sleeping {wait}s")
            time.sleep(wait)
    return []


def collect(session, kind, jobs, size, sleep, resume) -> tuple[list[dict], int, int]:
    """Walk a list of (label, lo, hi) jobs, checkpointing each one as it lands."""
    rows: list[dict] = []
    reused = fetched = 0
    bar = tqdm(jobs, desc=kind, unit="stratum")
    for label, lo, hi in bar:
        path = shard_path(kind, label, lo)
        if resume and path.exists():
            rows.extend(read_shard(path))
            reused += 1
            continue
        got = query(session, size, lo, hi, label, sleep)
        write_shard(path, got)
        rows.extend(got)
        fetched += 1
    bar.close()
    return rows, reused, fetched


def tidy(rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=KEEP_FIELDS)
    df = pd.DataFrame(rows)
    for col in KEEP_FIELDS:
        if col not in df.columns:
            df[col] = None
    df = df[KEEP_FIELDS]
    df["date_received"] = pd.to_datetime(df["date_received"], errors="coerce", utc=True)
    df = df.dropna(subset=["complaint_what_happened"])
    df = df[df["complaint_what_happened"].str.strip().str.len() > 0]
    return df.drop_duplicates(subset="complaint_id").reset_index(drop=True)


def checksum(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------- main

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", default="2023-01", help="first month, YYYY-MM")
    ap.add_argument("--end", default="2024-12", help="last month, YYYY-MM")
    ap.add_argument("--per-stratum", type=int, default=75,
                    help="max narratives per product per month")
    ap.add_argument("--per-month", type=int, default=825,
                    help="narratives per month for the natural distribution sample")
    ap.add_argument("--sleep", type=float, default=1.0,
                    help="seconds to pause between API calls, to stay under the rate limit")
    ap.add_argument("--fresh", action="store_true",
                    help="ignore existing checkpoints and refetch every stratum")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit without calling the API")
    args = ap.parse_args()

    windows = month_windows(args.start, args.end)
    strat_jobs = [(p, lo, hi) for lo, hi in windows for p in PRODUCTS]
    nat_jobs = [(None, lo, hi) for lo, hi in windows]
    total_requests = len(strat_jobs) + len(nat_jobs)
    done = sum(1 for k, jobs in (("stratified", strat_jobs), ("natural", nat_jobs))
               for lab, lo, _ in jobs if shard_path(k, lab, lo).exists()) if not args.fresh else 0

    print(f"months          : {len(windows)}  ({args.start} to {args.end})")
    print(f"products        : {len(PRODUCTS)}")
    print(f"stratified cap  : {args.per_stratum} per stratum, "
          f"upper bound {len(strat_jobs) * args.per_stratum:,} rows")
    print(f"natural sample  : {args.per_month} per month, "
          f"upper bound {len(nat_jobs) * args.per_month:,} rows")
    print(f"requests        : {total_requests} total, {done} already checkpointed, "
          f"{total_requests - done} to fetch")
    print(f"throttle        : {args.sleep}s between calls, "
          f"roughly {(total_requests - done) * (args.sleep + 0.7) / 60:.1f} min remaining")

    if args.dry_run:
        print("\ndry run, nothing fetched")
        return 0

    if args.fresh and SHARDS.exists():
        shutil.rmtree(SHARDS)
    SHARDS.mkdir(parents=True, exist_ok=True)
    RAW.mkdir(parents=True, exist_ok=True)

    # Do not set a custom User-Agent here. The consumerfinance.gov edge returns 403 for both
    # custom agent strings and browser-like ones; the stock requests agent is accepted.
    session = requests.Session()

    started = time.time()
    try:
        strat_rows, sr, sf = collect(session, "stratified", strat_jobs,
                                     args.per_stratum, args.sleep, not args.fresh)
        nat_rows, nr, nf = collect(session, "natural", nat_jobs,
                                   args.per_month, args.sleep, not args.fresh)
    except Exception as exc:  # noqa: BLE001
        kept = len(list(SHARDS.glob("*.jsonl")))
        print(f"\nStopped: {exc.__class__.__name__}: {exc}", file=sys.stderr)
        print(f"{kept} strata are checkpointed and safe. Rerun the same command to resume "
              f"from there; nothing already fetched is refetched.", file=sys.stderr)
        return 1

    strat, nat = tidy(strat_rows), tidy(nat_rows)
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
        "shards": {"stratified_reused": sr, "stratified_fetched": sf,
                   "natural_reused": nr, "natural_fetched": nf},
        "files": {
            "stratified": {"path": strat_path.name, "rows": int(len(strat)),
                           "sha256": checksum(strat_path),
                           "product_counts": strat["product"].value_counts().to_dict()},
            "natural": {"path": nat_path.name, "rows": int(len(nat)),
                        "sha256": checksum(nat_path),
                        "product_counts": nat["product"].value_counts().to_dict()},
        },
    }
    (RAW / "fetch_metadata.json").write_text(json.dumps(meta, indent=2))

    print(f"\nstratified : {len(strat):,} rows -> {strat_path}")
    print(f"natural    : {len(nat):,} rows -> {nat_path}")
    print(f"metadata   : {RAW / 'fetch_metadata.json'}")
    print(f"elapsed    : {meta['elapsed_seconds'] / 60:.1f} min")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
