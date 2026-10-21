"""Watchlist and project-wide settings.

All tickers are US-listed (ordinary shares or ADRs), so every stock shares
one trading calendar and one market close (16:00 New York time). That keeps
the "which trading day does this headline belong to?" rule simple and the
same for every stock.
"""

from pathlib import Path

# Folder paths, relative to the repository root.
ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
HEADLINES_DIR = DATA_DIR / "headlines"   # one CSV per month, e.g. 2026-10.csv
PRICES_FILE = DATA_DIR / "prices.csv"

# Benchmark used to compute market-adjusted (abnormal) returns.
MARKET_TICKER = "SPY"

# ticker -> (search name, keywords a headline must contain to count as relevant)
WATCHLIST: dict[str, tuple[str, list[str]]] = {
    "AAPL": ("Apple", ["apple", "aapl", "iphone"]),
    "MSFT": ("Microsoft", ["microsoft", "msft"]),
    "NVDA": ("Nvidia", ["nvidia", "nvda"]),
    "AMZN": ("Amazon", ["amazon", "amzn", "aws"]),
    "TSLA": ("Tesla", ["tesla", "tsla"]),
    "JPM": ("JPMorgan", ["jpmorgan", "jpm", "jp morgan"]),
    "XOM": ("ExxonMobil", ["exxon", "xom"]),
    "SHEL": ("Shell", ["shell", "shel"]),
    "TM": ("Toyota", ["toyota"]),
    "BABA": ("Alibaba", ["alibaba", "baba"]),
}

# FinBERT model on Hugging Face (Araci, 2019; fine-tuned by Prosus AI).
MODEL_NAME = "ProsusAI/finbert"

# US market close in New York time. News published at or after this time
# can only affect prices on the NEXT trading day.
MARKET_CLOSE_HOUR = 16
MARKET_TZ = "America/New_York"
