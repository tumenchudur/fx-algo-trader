"""
Parse BIS central bank policy rates from /tmp/cbpol.csv into a clean per-currency
daily-resampled rate table saved at data/rates/short_rates.parquet.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

OUT = Path("data/rates"); OUT.mkdir(parents=True, exist_ok=True)

# BIS uses ISO-3 country codes; XM is the euro area
CCY_TO_BIS = {
    "USD": "US",
    "EUR": "XM",
    "GBP": "GB",
    "JPY": "JP",
    "AUD": "AU",
    "NZD": "NZ",
    "CAD": "CA",
    "CHF": "CH",
}


def main() -> None:
    src = "/tmp/cbpol.csv"
    print("Reading BIS CBPOL CSV...")
    df = pd.read_csv(src, low_memory=False)
    print(f"  total rows: {len(df):,}")
    print(f"  columns: {list(df.columns)[:8]}")

    # Filter to monthly frequency (M)
    freq = df["FREQ:Frequency"].astype(str).str.startswith("M")
    df = df[freq].copy()
    # REF_AREA looks like "US: United States" — extract code
    df["country_code"] = df["REF_AREA:Reference area"].astype(str).str.split(":").str[0].str.strip()

    out = []
    for ccy, code in CCY_TO_BIS.items():
        sub = df[df["country_code"] == code]
        if sub.empty:
            print(f"  {ccy}: NO DATA for code {code}")
            continue
        sub = sub[["TIME_PERIOD:Time period or range", "OBS_VALUE:Observation Value"]].copy()
        sub.columns = ["period", "rate"]
        sub["date"] = pd.to_datetime(sub["period"], errors="coerce")
        sub = sub.dropna(subset=["date"])
        sub["rate"] = pd.to_numeric(sub["rate"], errors="coerce")
        sub = sub.dropna(subset=["rate"])
        sub = sub.set_index("date").sort_index()
        s = sub["rate"].rename(ccy)
        # Tz-localize to UTC for downstream join with FX
        s.index = s.index.tz_localize("UTC")
        print(f"  {ccy}: rows={len(s):>4d}  {s.index.min().date()} -> {s.index.max().date()}  last={s.iloc[-1]:.2f}%")
        out.append(s)

    combined = pd.concat(out, axis=1).sort_index()
    daily = combined.resample("1D").ffill()
    daily = daily.dropna(how="all")
    daily.to_parquet(OUT / "short_rates.parquet")
    print(f"\nSaved daily rates: {daily.shape} -> {OUT / 'short_rates.parquet'}")
    print("\nLast 5 rows:")
    print(daily.tail(5).round(2).to_string())
    print("\nSample 2010-01:")
    print(daily.loc["2010-01-01":"2010-01-05"].round(2).to_string())


if __name__ == "__main__":
    main()
