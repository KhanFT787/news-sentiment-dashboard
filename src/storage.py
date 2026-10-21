"""Read and write the project's CSV data files.

Headlines are stored one file per month (data/headlines/2026-10.csv). The
daily job then only rewrites the current month's file, which keeps each Git
commit small instead of rewriting one ever-growing file every day.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import HEADLINES_DIR, PRICES_FILE


def load_headlines(folder: Path = HEADLINES_DIR) -> pd.DataFrame:
    """Load every monthly headline file into one DataFrame."""
    files = sorted(Path(folder).glob("*.csv"))
    if not files:
        return pd.DataFrame()
    df = pd.concat((pd.read_csv(f) for f in files), ignore_index=True)
    df["published_utc"] = pd.to_datetime(df["published_utc"], utc=True, format="ISO8601")
    return df


def save_headlines(df: pd.DataFrame, folder: Path = HEADLINES_DIR) -> list[Path]:
    """Write headlines into monthly files (overwriting those months). Returns the files written."""
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    month = df["published_utc"].dt.strftime("%Y-%m")
    written = []
    for key, part in df.groupby(month):
        path = folder / f"{key}.csv"
        part.sort_values(["published_utc", "ticker"]).to_csv(path, index=False, float_format="%.4f")
        written.append(path)
    return written


def load_prices(path: Path = PRICES_FILE) -> pd.DataFrame:
    """Load the long price table (date, ticker, close)."""
    if not Path(path).exists():
        return pd.DataFrame(columns=["date", "ticker", "close"])
    df = pd.read_csv(path, parse_dates=["date"])
    return df


def save_prices(df: pd.DataFrame, path: Path = PRICES_FILE) -> None:
    """Write the long price table."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, float_format="%.4f", date_format="%Y-%m-%d")
