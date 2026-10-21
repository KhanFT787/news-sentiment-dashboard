"""Link daily news sentiment to stock returns.

Steps
-----
1. Assign every headline to the trading day whose price it can affect.
2. Average headline scores into one sentiment value per stock per day.
3. Compute daily returns and market-adjusted (abnormal) returns:
       r_t          = P_t / P_(t-1) - 1
       abnormal r_t = r_t(stock) - r_t(SPY)
   Subtracting the market strips out moves that hit every stock at once
   (a Fed decision, say), leaving the stock-specific part news could explain.
4. Measure how sentiment on day t relates to abnormal returns on
   day t-1 (did news follow the price?), day t (same day) and day t+1
   (does news predict tomorrow?).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from src.config import MARKET_CLOSE_HOUR, MARKET_TICKER, MARKET_TZ


# ---------------------------------------------------------------- step 1
def assign_trading_day(published_utc: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Map each publication time to the first trading day the news could affect.

    Rule: convert to New York time. News at or after the 16:00 close counts
    for the next calendar day. Weekend and holiday dates then roll forward to
    the next trading day. News newer than the last known trading day gets NaT
    (it has no price yet).
    """
    local = pd.to_datetime(published_utc, utc=True).dt.tz_convert(MARKET_TZ)
    day = local.dt.tz_localize(None).dt.normalize()
    after_close = local.dt.hour >= MARKET_CLOSE_HOUR
    day = day + pd.to_timedelta(after_close.astype(int), unit="D")

    trading_days = pd.DatetimeIndex(sorted(trading_days.unique()))
    pos = trading_days.searchsorted(day.values, side="left")
    result = pd.Series(pd.NaT, index=published_utc.index, dtype="datetime64[ns]")
    valid = pos < len(trading_days)
    result[valid] = trading_days[pos[valid]]
    return result


# ---------------------------------------------------------------- step 2
def daily_sentiment(scored: pd.DataFrame) -> pd.DataFrame:
    """One row per (ticker, trading_day): mean score, headline count, % positive, % negative."""
    df = scored.dropna(subset=["trading_day"])
    grouped = df.groupby(["ticker", "trading_day"])
    out = grouped.agg(
        sentiment=("score", "mean"),
        n_headlines=("score", "size"),
        pct_positive=("label", lambda s: (s == "positive").mean()),
        pct_negative=("label", lambda s: (s == "negative").mean()),
    )
    return out.reset_index().rename(columns={"trading_day": "date"})


# ---------------------------------------------------------------- step 3
def compute_returns(prices: pd.DataFrame, market: str = MARKET_TICKER) -> pd.DataFrame:
    """From a long price table (date, ticker, close) build daily, abnormal, previous- and next-day returns."""
    wide = prices.pivot(index="date", columns="ticker", values="close").sort_index()
    rets = wide.pct_change(fill_method=None)
    abnormal = rets.sub(rets[market], axis=0)

    # Each wide table (dates x tickers) becomes one column of a long table.
    pieces = {
        "ret": rets,
        "abn_ret": abnormal,
        "abn_ret_prev": abnormal.shift(1),   # yesterday's abnormal return
        "abn_ret_next": abnormal.shift(-1),  # tomorrow's abnormal return
    }
    long = pd.concat({name: table.stack() for name, table in pieces.items()}, axis=1)
    long.index.names = ["date", "ticker"]
    long = long.reset_index()
    return long[long["ticker"] != market].reset_index(drop=True)


def build_panel(daily: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    """Join daily sentiment with returns on (ticker, date). Days without news are dropped."""
    return daily.merge(returns, on=["ticker", "date"], how="inner").sort_values(["ticker", "date"])


# ---------------------------------------------------------------- step 4
HORIZONS = {
    "Previous day (t-1)": "abn_ret_prev",
    "Same day (t)": "abn_ret",
    "Next day (t+1)": "abn_ret_next",
}


def correlation_table(panel: pd.DataFrame, min_headlines: int = 1) -> pd.DataFrame:
    """Pearson and Spearman correlation between daily sentiment and abnormal returns at each horizon.

    Only days with at least `min_headlines` headlines are used: a day with a
    single headline gives a much noisier sentiment reading.
    """
    data = panel[panel["n_headlines"] >= min_headlines]
    rows = []
    for label, col in HORIZONS.items():
        pair = data[["sentiment", col]].dropna()
        n = len(pair)
        if n >= 3 and pair["sentiment"].nunique() > 1 and pair[col].nunique() > 1:
            r, p = stats.pearsonr(pair["sentiment"], pair[col])
            rho, _ = stats.spearmanr(pair["sentiment"], pair[col])
        else:
            r = p = rho = np.nan
        rows.append({"horizon": label, "pearson_r": r, "p_value": p, "spearman_rho": rho, "n_days": n})
    return pd.DataFrame(rows)


def bucket_table(panel: pd.DataFrame, min_headlines: int = 1) -> pd.DataFrame:
    """Split days into thirds by sentiment and show the average abnormal return in each group.

    A simple, intuitive check: if news predicts returns, the most-positive
    third of days should be followed by higher returns than the most-negative third.
    """
    data = panel[panel["n_headlines"] >= min_headlines].dropna(subset=["sentiment"]).copy()
    if data["sentiment"].nunique() < 3:
        return pd.DataFrame(columns=["bucket", "days", "avg_sentiment", "same_day", "next_day"])
    data["bucket"] = pd.qcut(
        data["sentiment"].rank(method="first"), 3, labels=["Most negative third", "Middle third", "Most positive third"]
    )
    out = data.groupby("bucket", observed=True).agg(
        days=("sentiment", "size"),
        avg_sentiment=("sentiment", "mean"),
        same_day=("abn_ret", "mean"),
        next_day=("abn_ret_next", "mean"),
    )
    return out.reset_index()


def describe_correlation(r: float, p: float) -> str:
    """Plain-English reading of a correlation coefficient and its p-value."""
    if np.isnan(r):
        return "Not enough data yet."
    size = abs(r)
    strength = "negligible" if size < 0.1 else "weak" if size < 0.3 else "moderate" if size < 0.5 else "strong"
    direction = "positive" if r > 0 else "negative"
    significance = "statistically significant at 5%" if p < 0.05 else "not statistically significant"
    return f"{strength.capitalize()} {direction} relationship ({significance})."
