"""
Download short-term interest rate data from FRED for the 8 G10 currencies.

For each currency we want a *short rate* that proxies the carry yield.
We use 3-month interbank rates where available (cleanest carry proxy);
fall back to central-bank policy / target rates otherwise.

FRED CSV download: no API key needed for the public CSV endpoint.
"""

from __future__ import annotations

import io
from pathlib import Path
from urllib.request import urlopen, Request

import pandas as pd

OUT = Path("data/rates"); OUT.mkdir(parents=True, exist_ok=True)

# Currency -> FRED series ID. Picked for: long history, daily/monthly, short-tenor.
SERIES = {
    "USD": "DGS3MO",       # 3-Month Treasury daily
    "EUR": "IR3TIB01EZM156N",  # 3-Month interbank EUR (monthly)
    "GBP": "IR3TIB01GBM156N",  # 3-Month interbank GBP (monthly)
    "JPY": "IR3TIB01JPM156N",  # 3-Month interbank JPY (monthly)
    "AUD": "IR3TIB01AUM156N",  # 3-Month interbank AUD (monthly)
    "NZD": "IR3TIB01NZM156N",  # 3-Month interbank NZD (monthly)
    "CAD": "IR3TIB01CAM156N",  # 3-Month interbank CAD (monthly)
    "CHF": "IR3TIB01CHM156N",  # 3-Month interbank CHF (monthly)
}


def fetch_fred(series: str) -> pd.Series:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series}"
    req = Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(req, timeout=30) as r:
        raw = r.read()
    df = pd.read_csv(io.BytesIO(raw))
    df.columns = ["date", "value"]
    df["date"] = pd.to_datetime(df["date"], utc=True)
    df = df.set_index("date").sort_index()
    df["value"] = pd.to_numeric(df["value"], errors="coerce")
    return df["value"].dropna().rename(series)


if __name__ == "__main__":
    out = []
    for ccy, sid in SERIES.items():
        try:
            s = fetch_fred(sid)
            print(f"  {ccy:3s} {sid:25s} rows={len(s):>5d}  {s.index.min().date()} -> {s.index.max().date()}  last={s.iloc[-1]:.3f}%")
            s.name = ccy
            out.append(s)
        except Exception as e:
            print(f"  {ccy} {sid} ERROR: {e}")

    df = pd.concat(out, axis=1).sort_index()
    # Forward-fill monthly series to daily
    df = df.resample("1D").ffill()
    df = df.dropna(how="all")
    df.to_parquet(OUT / "short_rates.parquet")
    print(f"\nSaved combined daily rates: {df.shape} -> {OUT / 'short_rates.parquet'}")
    print(df.tail(5).round(3).to_string())
