"""Collect raw data: news headlines (Google News RSS) and daily prices (Yahoo Finance).

Nothing in this module does any analysis — it only downloads and tidies data
so the rest of the project can work with clean pandas DataFrames.
"""

from __future__ import annotations

import re
import time
from datetime import date
from urllib.parse import quote_plus

import feedparser
import pandas as pd

GOOGLE_NEWS_RSS = "https://news.google.com/rss/search?q={query}&hl=en-US&gl=US&ceid=US:en"

HEADLINE_COLUMNS = ["published_utc", "ticker", "headline", "source", "link"]


def build_query(name: str, start: date | None = None, end: date | None = None) -> str:
    """Build a Google News search query such as '"Apple" stock after:2026-07-01 before:2026-07-08'.

    With no dates, the query asks for the last 2 days (used by the daily job).
    The after/before operators let us backfill history one week at a time.
    """
    query = f'"{name}" stock'
    if start and end:
        query += f" after:{start.isoformat()} before:{end.isoformat()}"
    else:
        query += " when:2d"
    return query


def clean_headline(title: str, source: str | None) -> str:
    """Remove the ' - Source Name' suffix Google News appends, and tidy whitespace."""
    title = title.strip()
    if source and title.endswith(f" - {source}"):
        title = title[: -len(f" - {source}")]
    return re.sub(r"\s+", " ", title).strip()


def is_relevant(headline: str, keywords: list[str]) -> bool:
    """Keep a headline only if it mentions the company (whole-word match on any keyword)."""
    text = headline.lower()
    return any(re.search(rf"\b{re.escape(k)}\b", text) for k in keywords)


def parse_feed(feed, ticker: str, keywords: list[str]) -> pd.DataFrame:
    """Turn a parsed RSS feed into a DataFrame of relevant, cleaned headlines."""
    rows = []
    for entry in feed.entries:
        source = entry.get("source", {}).get("title")
        headline = clean_headline(entry.get("title", ""), source)
        if not headline or not is_relevant(headline, keywords):
            continue
        rows.append(
            {
                "published_utc": pd.to_datetime(entry.get("published"), utc=True, errors="coerce"),
                "ticker": ticker,
                "headline": headline,
                "source": source,
                "link": entry.get("link"),
            }
        )
    df = pd.DataFrame(rows, columns=HEADLINE_COLUMNS)
    return df.dropna(subset=["published_utc"])


def fetch_headlines(
    ticker: str,
    name: str,
    keywords: list[str],
    start: date | None = None,
    end: date | None = None,
) -> pd.DataFrame:
    """Download headlines for one company from Google News RSS (free, no API key)."""
    url = GOOGLE_NEWS_RSS.format(query=quote_plus(build_query(name, start, end)))
    feed = feedparser.parse(url)
    return parse_feed(feed, ticker, keywords)


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """Drop repeated stories: same ticker and same headline text (case-insensitive).

    The same wire story is often syndicated by several outlets; counting it
    five times would overweight it in the daily average.
    """
    key = df["ticker"] + "|" + df["headline"].str.lower().str.strip()
    return df.loc[~key.duplicated()].reset_index(drop=True)


def fetch_prices(tickers: list[str], start: date, end: date | None = None, retries: int = 3) -> pd.DataFrame:
    """Download daily adjusted close prices from Yahoo Finance via yfinance.

    Returns a long table with columns: date, ticker, close.
    """
    import yfinance as yf  # imported here so the app and tests don't need it

    closes = pd.DataFrame()
    for attempt in range(retries):
        try:
            raw = yf.download(
                tickers, start=start, end=end, auto_adjust=True, progress=False, group_by="column"
            )
            closes = raw["Close"]
            if isinstance(closes, pd.Series):  # single ticker
                closes = closes.to_frame(tickers[0])
            if not closes.dropna(how="all").empty:
                break
        except Exception:  # network hiccup or rate limit: wait and retry
            closes = pd.DataFrame()
        time.sleep(5 * (attempt + 1))

    if closes.dropna(how="all").empty:
        raise RuntimeError("Could not download prices from Yahoo Finance.")

    long = closes.reset_index().melt(id_vars="Date", var_name="ticker", value_name="close")
    long = long.rename(columns={"Date": "date"}).dropna(subset=["close"])
    long["date"] = pd.to_datetime(long["date"]).dt.tz_localize(None).dt.normalize()
    return long.sort_values(["ticker", "date"]).reset_index(drop=True)
