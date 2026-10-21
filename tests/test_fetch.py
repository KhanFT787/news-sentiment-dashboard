"""Tests for headline cleaning, filtering and de-duplication (no internet needed)."""

from datetime import date
from types import SimpleNamespace

import pandas as pd

from src.fetch import build_query, clean_headline, deduplicate, is_relevant, parse_feed


def test_build_query_daily_and_backfill():
    assert build_query("Apple") == '"Apple" stock when:2d'
    q = build_query("Apple", date(2026, 7, 1), date(2026, 7, 8))
    assert q == '"Apple" stock after:2026-07-01 before:2026-07-08'


def test_clean_headline_strips_source_suffix():
    assert clean_headline("Apple beats estimates - CNBC", "CNBC") == "Apple beats estimates"
    assert clean_headline("  Apple   beats  ", None) == "Apple beats"


def test_is_relevant_uses_whole_words():
    assert is_relevant("Tesla shares jump", ["tesla", "tsla"])
    assert not is_relevant("Markets rally on Fed", ["tesla", "tsla"])
    assert not is_relevant("Pineapple prices rise", ["apple"])  # 'apple' inside another word


def test_parse_feed_keeps_only_relevant_entries():
    feed = SimpleNamespace(
        entries=[
            {"title": "Apple hits record - Reuters", "source": {"title": "Reuters"},
             "published": "Tue, 20 Oct 2026 15:19:01 GMT", "link": "https://x"},
            {"title": "Oil falls - Reuters", "source": {"title": "Reuters"},
             "published": "Tue, 20 Oct 2026 15:19:01 GMT", "link": "https://y"},
        ]
    )
    df = parse_feed(feed, "AAPL", ["apple"])
    assert list(df["headline"]) == ["Apple hits record"]
    assert df["published_utc"].iloc[0] == pd.Timestamp("2026-10-20 15:19:01", tz="UTC")


def test_deduplicate_is_case_insensitive_and_per_ticker():
    df = pd.DataFrame(
        {"ticker": ["AAPL", "AAPL", "MSFT"], "headline": ["Big News", "big news", "Big News"]}
    )
    out = deduplicate(df)
    assert len(out) == 2
    assert set(out["ticker"]) == {"AAPL", "MSFT"}
