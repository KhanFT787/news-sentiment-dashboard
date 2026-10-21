"""Streamlit dashboard: financial news sentiment vs stock price moves.

Run locally:  streamlit run app.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from src.analysis import (
    assign_trading_day,
    bucket_table,
    build_panel,
    compute_returns,
    correlation_table,
    daily_sentiment,
    describe_correlation,
)
from src.config import WATCHLIST
from src.storage import load_headlines, load_prices

st.set_page_config(page_title="News Sentiment Dashboard", page_icon="📰", layout="wide")

POS, NEG, NEU, INK = "#1a9e77", "#d6543c", "#8a8f98", "#2b59c3"
DISCLAIMER = "Educational purposes only. Not investment advice."


# ------------------------------------------------------------------ data
@st.cache_data(ttl=3600)
def load_all() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load stored data and build the analysis panel (cached for an hour)."""
    headlines = load_headlines()
    prices = load_prices()
    trading_days = pd.DatetimeIndex(prices["date"].unique())
    headlines["trading_day"] = assign_trading_day(headlines["published_utc"], trading_days)
    panel = build_panel(daily_sentiment(headlines), compute_returns(prices))
    return headlines, prices, panel


@st.cache_resource(show_spinner="Loading FinBERT (first time takes about a minute)...")
def get_model():
    from src.sentiment import load_finbert

    return load_finbert()


headlines, prices, panel = load_all()

# ------------------------------------------------------------------ sidebar
with st.sidebar:
    st.header("Settings")
    ticker = st.selectbox(
        "Stock", list(WATCHLIST), format_func=lambda t: f"{WATCHLIST[t][0]} ({t})"
    )
    first, last = panel["date"].min().date(), panel["date"].max().date()
    start, end = st.slider("Period", min_value=first, max_value=last, value=(first, last), format="MMM YYYY")
    min_n = st.slider(
        "Minimum headlines per day", 1, 5, 2,
        help="Days with fewer headlines are left out of the statistics. More headlines = a less noisy daily score.",
    )
    pooled = st.toggle("Pool all 10 stocks for statistics", value=False,
                       help="Uses every stock's days together: more data, so more reliable estimates.")
    st.divider()
    st.caption(f"Headlines: Google News RSS · Prices: Yahoo Finance · Model: FinBERT\n\n{DISCLAIMER}")

in_period = lambda df, col: df[(df[col].dt.date >= start) & (df[col].dt.date <= end)]  # noqa: E731
stock_panel = in_period(panel[panel["ticker"] == ticker], "date")
stock_news = headlines[(headlines["ticker"] == ticker)].dropna(subset=["trading_day"])
stock_news = in_period(stock_news, "trading_day")
stock_prices = in_period(prices[prices["ticker"] == ticker], "date")
stats_panel = in_period(panel, "date") if pooled else stock_panel

# ------------------------------------------------------------------ header
name = WATCHLIST[ticker][0]
st.title("📰 Financial News Sentiment Dashboard")
st.markdown(
    "Does the tone of financial news line up with how a stock moves? Each headline is scored by "
    "**FinBERT**, an AI model trained on financial text, and compared with the stock's "
    "**market-adjusted return**."
)
st.caption(f"Data up to {headlines['published_utc'].max():%d %b %Y} · updated daily · {DISCLAIMER}")

c1, c2, c3, c4 = st.columns(4)
c1.metric("Headlines", f"{len(stock_news):,}")
c2.metric("Average sentiment", f"{stock_news['score'].mean():+.2f}", help="-1 = all negative, +1 = all positive")
c3.metric("Positive / negative", f"{(stock_news['label'] == 'positive').mean():.0%} / {(stock_news['label'] == 'negative').mean():.0%}")
c4.metric("Days with news", f"{len(stock_panel):,}")

# ------------------------------------------------------------------ price vs sentiment chart
st.subheader(f"{name}: price and daily news sentiment")
fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.6, 0.4], vertical_spacing=0.06)
fig.add_trace(go.Scatter(x=stock_prices["date"], y=stock_prices["close"], name="Price (USD)",
                         line=dict(color=INK, width=2)), row=1, col=1)
colors = np.where(stock_panel["sentiment"] >= 0, POS, NEG)
fig.add_trace(go.Bar(x=stock_panel["date"], y=stock_panel["sentiment"], name="Daily sentiment",
                     marker_color=colors, opacity=0.55,
                     customdata=stock_panel["n_headlines"],
                     hovertemplate="%{x|%d %b %Y}<br>Sentiment %{y:+.2f}<br>%{customdata} headlines<extra></extra>"),
              row=2, col=1)
smooth = stock_panel.set_index("date")["sentiment"].rolling("14D").mean()
fig.add_trace(go.Scatter(x=smooth.index, y=smooth.values, name="14-day average",
                         line=dict(color="#222", width=1.5)), row=2, col=1)
fig.update_yaxes(title_text="Price (USD)", row=1, col=1)
fig.update_yaxes(title_text="Sentiment", range=[-1, 1], row=2, col=1)
fig.update_layout(height=520, margin=dict(l=10, r=10, t=10, b=10), hovermode="x unified",
                  legend=dict(orientation="h", y=1.04, x=0))
st.plotly_chart(fig, use_container_width=True)

# ------------------------------------------------------------------ statistics
scope = "all 10 stocks pooled" if pooled else name
st.subheader(f"Does sentiment line up with returns? ({scope})")
st.markdown(
    "Correlation between a day's average sentiment and the stock's **abnormal return** "
    "(stock return minus S&P 500 ETF return) on the previous day, the same day and the next day."
)
corr = correlation_table(stats_panel, min_headlines=min_n)
left, right = st.columns([3, 2])
with left:
    for _, row in corr.iterrows():
        r = row["pearson_r"]
        value = "n/a" if np.isnan(r) else f"r = {r:+.2f}, p = {row['p_value']:.3f}"
        st.markdown(f"**{row['horizon']}** — {value} · {int(row['n_days'])} days  \n"
                    f"{describe_correlation(r, row['p_value'])}")
    with st.expander("How to read this"):
        st.markdown(
            "- **Same day:** news and price moving together is expected — the news often *reports* the move.\n"
            "- **Previous day:** a link here suggests news *follows* prices (journalists explain yesterday's move).\n"
            "- **Next day:** a link here would mean news *predicts* returns. In an efficient market this should "
            "be close to zero, because public news is priced in quickly.\n"
            "- **p-value:** below 0.05 means a relationship this strong would be unlikely if there were truly none. "
            "Correlation is not causation, and many tests on the same data will produce some false positives."
        )
with right:
    buckets = bucket_table(stats_panel, min_headlines=min_n)
    if not buckets.empty:
        bfig = go.Figure()
        bfig.add_trace(go.Bar(x=buckets["bucket"], y=buckets["same_day"] * 100, name="Same day", marker_color=INK))
        bfig.add_trace(go.Bar(x=buckets["bucket"], y=buckets["next_day"] * 100, name="Next day", marker_color=NEU))
        bfig.update_layout(title="Average abnormal return by sentiment group", barmode="group", height=330,
                           yaxis_title="Abnormal return (%)", margin=dict(l=10, r=10, t=40, b=10),
                           legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(bfig, use_container_width=True)

# ------------------------------------------------------------------ headlines table
st.subheader(f"Latest {name} headlines")
label_filter = st.multiselect("Show", ["positive", "neutral", "negative"], default=["positive", "neutral", "negative"])
table = stock_news[stock_news["label"].isin(label_filter)].sort_values("published_utc", ascending=False)
st.dataframe(
    table[["published_utc", "headline", "source", "label", "score", "link"]].head(300),
    hide_index=True, use_container_width=True,
    column_config={
        "published_utc": st.column_config.DatetimeColumn("Published (UTC)", format="D MMM YYYY, HH:mm"),
        "headline": st.column_config.TextColumn("Headline", width="large"),
        "score": st.column_config.ProgressColumn("Score", min_value=-1, max_value=1, format="%+.2f"),
        "link": st.column_config.LinkColumn("Link", display_text="open"),
    },
)

# ------------------------------------------------------------------ try it yourself
with st.expander("🧪 Try it yourself: score any headline with FinBERT"):
    text = st.text_input("Headline", "Company beats earnings expectations and raises full-year guidance")
    if st.button("Score headline"):
        from src.sentiment import score_texts

        result = score_texts([text], classifier=get_model()).iloc[0]
        a, b, c, d = st.columns(4)
        a.metric("Label", result["label"].capitalize())
        b.metric("Positive", f"{result['p_positive']:.0%}")
        c.metric("Negative", f"{result['p_negative']:.0%}")
        d.metric("Score", f"{result['score']:+.2f}")

# ------------------------------------------------------------------ method
with st.expander("Methodology and limitations"):
    st.markdown(
        """
**Method**
1. Headlines for each stock are collected daily from Google News RSS and filtered to those naming the company.
   Duplicates (the same story syndicated by several outlets) are removed.
2. FinBERT gives each headline P(positive), P(negative), P(neutral). **Score = P(positive) − P(negative)**, from −1 to +1.
3. Each headline is assigned to the first trading day it could affect: news after the 16:00 New York close,
   or on a weekend or holiday, counts for the next trading day.
4. Daily sentiment = average score of that day's headlines.
5. Abnormal return = stock return − SPY return, to remove market-wide moves.

**Limitations**
- Headlines only, not full articles; Google News coverage varies by stock and over time.
- Backfilled history has date-level timestamps only, so some after-close news may be assigned to the wrong day.
- FinBERT can misread sarcasm, questions ("Should you buy...?") and company-irrelevant mentions.
- The watchlist is ten large, heavily covered companies, so results do not generalise to small caps.
- Correlation is not causation. This is an exploratory tool, not a trading signal.
"""
    )
st.caption(f"Built by Mehedi Hasan Khan · [GitHub](https://github.com/KhanFT787/news-sentiment-dashboard) · {DISCLAIMER}")
