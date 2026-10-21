"""Tests for trading-day assignment, returns and correlation logic (synthetic data only)."""

import numpy as np
import pandas as pd
import pytest

from src.analysis import (
    assign_trading_day,
    bucket_table,
    build_panel,
    compute_returns,
    correlation_table,
    daily_sentiment,
    describe_correlation,
)

# Mon 19 Oct to Fri 23 Oct 2026, then Mon 26 Oct (no weekend days).
TRADING_DAYS = pd.DatetimeIndex(pd.bdate_range("2026-10-19", "2026-10-26"))


def test_news_before_close_counts_same_day():
    # 14:00 UTC = 10:00 New York (summer time) -> same day
    t = pd.Series(pd.to_datetime(["2026-10-20 14:00"], utc=True))
    assert assign_trading_day(t, TRADING_DAYS).iloc[0] == pd.Timestamp("2026-10-20")


def test_news_after_close_counts_next_day():
    # 21:00 UTC = 17:00 New York -> after the 16:00 close -> next trading day
    t = pd.Series(pd.to_datetime(["2026-10-20 21:00"], utc=True))
    assert assign_trading_day(t, TRADING_DAYS).iloc[0] == pd.Timestamp("2026-10-21")


def test_weekend_news_rolls_to_monday():
    t = pd.Series(pd.to_datetime(["2026-10-24 15:00"], utc=True))  # Saturday
    assert assign_trading_day(t, TRADING_DAYS).iloc[0] == pd.Timestamp("2026-10-26")


def test_news_after_last_trading_day_is_unassigned():
    t = pd.Series(pd.to_datetime(["2026-10-27 15:00"], utc=True))
    assert pd.isna(assign_trading_day(t, TRADING_DAYS).iloc[0])


def test_daily_sentiment_averages_scores():
    scored = pd.DataFrame(
        {
            "ticker": ["AAPL"] * 3,
            "trading_day": pd.to_datetime(["2026-10-20"] * 2 + ["2026-10-21"]),
            "score": [0.6, -0.2, 0.5],
            "label": ["positive", "negative", "positive"],
        }
    )
    out = daily_sentiment(scored)
    first = out.iloc[0]
    assert first["sentiment"] == pytest.approx(0.2)
    assert first["n_headlines"] == 2
    assert first["pct_positive"] == pytest.approx(0.5)


def make_prices():
    dates = pd.bdate_range("2026-10-19", periods=4)
    return pd.DataFrame(
        {
            "date": list(dates) * 2,
            "ticker": ["AAPL"] * 4 + ["SPY"] * 4,
            "close": [100, 110, 110, 99, 100, 105, 105, 105],
        }
    )


def test_compute_returns_and_abnormal_returns():
    r = compute_returns(make_prices()).set_index("date")
    assert set(r["ticker"]) == {"AAPL"}  # market itself removed
    day2 = r.loc[pd.Timestamp("2026-10-20")]
    assert day2["ret"] == pytest.approx(0.10)
    assert day2["abn_ret"] == pytest.approx(0.10 - 0.05)
    assert day2["abn_ret_next"] == pytest.approx(0.0)
    assert day2["abn_ret_prev"] != day2["abn_ret_prev"]  # NaN: no return before day 1


def synthetic_panel(n=200, link=0.5, seed=0):
    """Panel where next-day return = link * sentiment + noise, so correlation should be positive."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(-1, 1, n)
    return pd.DataFrame(
        {
            "sentiment": s,
            "n_headlines": rng.integers(1, 6, n),
            "abn_ret": rng.normal(0, 0.01, n),
            "abn_ret_prev": rng.normal(0, 0.01, n),
            "abn_ret_next": link * 0.01 * s + rng.normal(0, 0.005, n),
        }
    )


def test_correlation_table_detects_planted_relationship():
    table = correlation_table(synthetic_panel()).set_index("horizon")
    nxt = table.loc["Next day (t+1)"]
    assert nxt["pearson_r"] > 0.4 and nxt["p_value"] < 0.01
    assert abs(table.loc["Same day (t)", "pearson_r"]) < 0.2  # pure noise


def test_correlation_table_min_headlines_filter():
    panel = synthetic_panel()
    table = correlation_table(panel, min_headlines=5)
    assert table["n_days"].iloc[0] == (panel["n_headlines"] >= 5).sum()


def test_correlation_with_too_little_data_is_nan():
    table = correlation_table(synthetic_panel(n=2))
    assert table["pearson_r"].isna().all()


def test_bucket_table_orders_returns():
    b = bucket_table(synthetic_panel()).set_index("bucket")
    assert b.loc["Most positive third", "next_day"] > b.loc["Most negative third", "next_day"]
    assert b["days"].sum() == 200


def test_build_panel_inner_join():
    daily = pd.DataFrame({"ticker": ["AAPL"], "date": [pd.Timestamp("2026-10-20")], "sentiment": [0.3], "n_headlines": [2]})
    panel = build_panel(daily, compute_returns(make_prices()))
    assert len(panel) == 1 and panel["abn_ret"].iloc[0] == pytest.approx(0.05)


def test_describe_correlation():
    assert describe_correlation(0.05, 0.6).startswith("Negligible positive")
    assert "statistically significant" in describe_correlation(-0.4, 0.001)
    assert describe_correlation(np.nan, np.nan) == "Not enough data yet."
