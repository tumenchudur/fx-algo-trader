"""
Download 15+ years of D1 spot FX from Yahoo Finance.

D1 history is available from ~2003 for FX majors via yfinance.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path("data/yf_d1"); OUT.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
}


def grab(symbol: str, ticker: str) -> None:
    print(f"  {symbol:8s} ({ticker}) ", end="", flush=True)
    try:
        df = yf.download(ticker, start="2005-01-01", end="2026-05-06",
                         interval="1d", auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        keep = [c for c in ["open","high","low","close","volume"] if c in df.columns]
        df = df[keep].dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        df.to_parquet(OUT / f"{symbol}.parquet")
        print(f"rows={len(df):>5d}  {df.index.min().date()} -> {df.index.max().date()}")
    except Exception as e:
        print(f"ERROR {e}")


if __name__ == "__main__":
    for s, t in TICKERS.items():
        grab(s, t)
