"""Daily data collection: fetch new headlines, score them with FinBERT, refresh prices.

Usage (from the repository root):
    python scripts/collect.py                    # daily update (last 2 days of news)
    python scripts/collect.py --backfill-days 365  # one-off: build a year of history

The GitHub Actions workflow in .github/workflows/daily.yml runs the daily
update automatically and commits the new data back to the repository.
"""

from __future__ import annotations

import argparse
import sys
import time
from datetime import date, timedelta
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))  # make `src` importable

from src.config import MARKET_TICKER, WATCHLIST  # noqa: E402
from src.fetch import deduplicate, fetch_headlines, fetch_prices  # noqa: E402
from src.sentiment import add_sentiment  # noqa: E402
from src.storage import load_headlines, save_headlines, save_prices  # noqa: E402


def week_windows(days_back: int, today: date) -> list[tuple[date, date]]:
    """Split the last `days_back` days into consecutive 7-day (after, before) search windows."""
    windows = []
    start = today - timedelta(days=days_back)
    while start < today:
        end = min(start + timedelta(days=7), today + timedelta(days=1))
        windows.append((start, end))
        start = start + timedelta(days=7)
    return windows


def collect_raw(backfill_days: int, today: date) -> pd.DataFrame:
    """Download headlines for every stock in the watchlist."""
    frames = []
    for ticker, (name, keywords) in WATCHLIST.items():
        if backfill_days:
            for start, end in week_windows(backfill_days, today):
                frames.append(fetch_headlines(ticker, name, keywords, start, end))
                time.sleep(1)  # be polite to the news server
        else:
            frames.append(fetch_headlines(ticker, name, keywords))
            time.sleep(1)
        print(f"  {ticker}: fetched")
    frames = [f for f in frames if not f.empty]
    return deduplicate(pd.concat(frames, ignore_index=True)) if frames else pd.DataFrame()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backfill-days", type=int, default=0, help="fetch this many past days of news")
    args = parser.parse_args()
    today = date.today()

    print("Fetching headlines...")
    existing = load_headlines()
    raw = collect_raw(args.backfill_days, today)

    if not raw.empty and not existing.empty:
        # Keep only headlines we have not stored before.
        seen = set(existing["ticker"] + "|" + existing["headline"].str.lower().str.strip())
        key = raw["ticker"] + "|" + raw["headline"].str.lower().str.strip()
        raw = raw.loc[~key.isin(seen)]

    print(f"New headlines to score: {len(raw)}")
    if not raw.empty:
        scored = add_sentiment(raw)
        combined = pd.concat([existing, scored], ignore_index=True) if not existing.empty else scored
        save_headlines(combined)

    print("Fetching prices...")
    all_headlines = load_headlines()
    first_day = all_headlines["published_utc"].min().date() - timedelta(days=10)
    prices = fetch_prices(list(WATCHLIST) + [MARKET_TICKER], start=first_day)
    save_prices(prices)
    print(f"Done. Stored {len(all_headlines)} headlines and {len(prices)} price rows.")


if __name__ == "__main__":
    main()
