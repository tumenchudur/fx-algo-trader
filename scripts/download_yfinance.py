"""
Download ~2 years of H1 data from Yahoo Finance for FX majors and cross-asset proxies.

yfinance hourly history is capped at 730 days. We grab:
- FX majors: EURUSD, GBPUSD, USDJPY, USDCHF, USDCAD, AUDUSD, NZDUSD
- USD index: DX-Y.NYB
- 10Y yield: ^TNX
- Equity vol: ^VIX
- Gold / oil: GC=F, CL=F
- Equity index: ^GSPC, ^NDX

Saved as parquet under data/yf/.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import yfinance as yf

OUT = Path("data/yf"); OUT.mkdir(parents=True, exist_ok=True)

TICKERS = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "USDCHF": "USDCHF=X",
    "USDCAD": "USDCAD=X",
    "AUDUSD": "AUDUSD=X",
    "NZDUSD": "NZDUSD=X",
    "DXY":    "DX-Y.NYB",
    "TNX":    "^TNX",     # 10Y yield x10
    "VIX":    "^VIX",
    "XAUUSD": "GC=F",
    "WTI":    "CL=F",
    "SPX":    "^GSPC",
    "NDX":    "^NDX",
}


def grab(symbol: str, ticker: str) -> None:
    print(f"  {symbol:8s} ({ticker}) ", end="", flush=True)
    try:
        df = yf.download(ticker, period="730d", interval="1h",
                         auto_adjust=False, progress=False, threads=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.rename(columns=str.lower)
        if df.empty:
            print("EMPTY"); return
        keep = [c for c in ["open", "high", "low", "close", "volume"] if c in df.columns]
        df = df[keep].dropna()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        else:
            df.index = df.index.tz_convert("UTC")
        df.to_parquet(OUT / f"{symbol}.parquet")
        print(f"rows={len(df):>5d}  {df.index.min().date()} -> {df.index.max().date()}")
    except Exception as e:
        print(f"ERROR {e}")


if __name__ == "__main__":
    for sym, tk in TICKERS.items():
        grab(sym, tk)
