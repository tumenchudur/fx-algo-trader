"""
Edge research at H1 timeframe + cross-asset signals.

Differences from edge_research.py:
- Aggregates available M5 data to H1 (and re-uses any native H1 if present).
- Tests longer horizons (4h, 1d, 3d, 1w).
- Adds calendar-effect signals (day-of-week, end-of-month, week-of-month).
- Adds a synthetic USD-strength index built from EURUSD/GBPUSD/AUDUSD/NZDUSD/USDJPY/USDCHF/USDCAD,
  then tests USD-momentum as an *exogenous* signal for each pair.
- Trades are non-overlapping. Costs identical to the M5 study.
"""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

COSTS_PIPS = {"EURUSD": 0.8, "GBPUSD": 1.0, "USDJPY": 1.0, "XAUUSD": 25.0,
              "AUDUSD": 1.0, "NZDUSD": 1.5, "USDCHF": 1.5, "USDCAD": 1.2}
PIP_SIZE = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "XAUUSD": 1e-2,
            "AUDUSD": 1e-4, "NZDUSD": 1e-4, "USDCHF": 1e-4, "USDCAD": 1e-4}
# In standard FX naming, "+1 long" means buy base / sell quote.
# Pairs where USD is the QUOTE (rises when USD weakens): EURUSD, GBPUSD, AUDUSD, NZDUSD, XAUUSD.
# Pairs where USD is the BASE (rises when USD strengthens): USDJPY, USDCHF, USDCAD.
USD_QUOTE = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "XAUUSD"}
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}


def load_h1() -> dict[str, pd.DataFrame]:
    """Load every M5/H1 parquet, aggregate M5 to H1, merge per-symbol."""
    files = sorted(
        glob.glob("data/sampl/*.parquet")
        + glob.glob("data/sample/*_M5.parquet")
        + glob.glob("data/sample/*_H1.parquet")
        + glob.glob("data/processed/*.parquet")
        + glob.glob("data/long_history/*.parquet")
    )
    bag: dict[str, list[pd.DataFrame]] = {}
    for f in files:
        sym = Path(f).stem.split("_")[0]
        df = pd.read_parquet(f)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        if not all(c in df.columns for c in ("open", "high", "low", "close")):
            continue
        df = df[["open", "high", "low", "close"]].astype(float)
        # Resample to H1 if not already
        if "_H1" not in f and "_h1" not in f and "H1" not in Path(f).stem:
            h1 = df.resample("1h").agg({
                "open": "first", "high": "max", "low": "min", "close": "last"
            }).dropna()
            df = h1
        bag.setdefault(sym, []).append(df)

    out = {}
    for sym, frames in bag.items():
        merged = pd.concat(frames).sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]
        # Need >=1000 bars to be useful
        if len(merged) >= 500:
            out[sym] = merged
    return out


def build_usd_index(data: dict[str, pd.DataFrame]) -> pd.Series:
    """Synthetic USD strength index: geo-mean of USD-base pairs / USD-quote pairs."""
    components = []
    for sym, df in data.items():
        if sym == "XAUUSD":
            continue  # gold isn't a USD-strength signal cleanly
        # Up = USD strengthens
        if sym in USD_BASE:
            components.append(np.log(df["close"]).rename(sym))
        elif sym in USD_QUOTE:
            components.append(-np.log(df["close"]).rename(sym))
    if not components:
        return pd.Series(dtype=float)
    aligned = pd.concat(components, axis=1).dropna()
    return aligned.mean(axis=1)


# ---------- indicators ----------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    d = close.diff()
    g = d.clip(lower=0).rolling(period).mean()
    l = (-d.clip(upper=0)).rolling(period).mean()
    rs = g / l.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def zscore(s: pd.Series, period: int) -> pd.Series:
    return (s - s.rolling(period).mean()) / s.rolling(period).std()


# ---------- signals ----------

def sig_momentum(df: pd.DataFrame, lookback: int) -> pd.Series:
    r = df["close"].pct_change(lookback)
    s = pd.Series(0, index=df.index, dtype=int)
    s[r > 0] = 1; s[r < 0] = -1
    return s


def sig_meanrev_z(df: pd.DataFrame, lookback: int, threshold: float) -> pd.Series:
    z = zscore(df["close"], lookback)
    s = pd.Series(0, index=df.index, dtype=int)
    s[z < -threshold] = 1; s[z > threshold] = -1
    return s


def sig_donchian(df: pd.DataFrame, lookback: int) -> pd.Series:
    up = df["high"].rolling(lookback).max().shift(1)
    dn = df["low"].rolling(lookback).min().shift(1)
    s = pd.Series(0, index=df.index, dtype=int)
    s[df["close"] > up] = 1; s[df["close"] < dn] = -1
    return s


def sig_dow(df: pd.DataFrame, long_dows: set[int], short_dows: set[int]) -> pd.Series:
    """Day-of-week. 0=Mon, 4=Fri. Fire only at first bar of day (00:00 UTC)."""
    dow = df.index.dayofweek
    h = df.index.hour
    first = h == 0
    s = pd.Series(0, index=df.index, dtype=int)
    s[first & np.isin(dow, list(long_dows))] = 1
    s[first & np.isin(dow, list(short_dows))] = -1
    return s


def sig_month_turn(df: pd.DataFrame, days_around: int = 2) -> pd.Series:
    """Long bias around month-end / month-start (well-documented FX flow)."""
    day = df.index.day
    h = df.index.hour
    # last `days_around` days of month or first `days_around`
    days_in_month = pd.Series(df.index.days_in_month, index=df.index)
    near_end = day > (days_in_month - days_around)
    near_start = day <= days_around
    first_bar = h == 0
    s = pd.Series(0, index=df.index, dtype=int)
    s[first_bar & (near_end | near_start)] = 1
    return s


def sig_usd_momentum(df: pd.DataFrame, usd_idx: pd.Series, symbol: str, lookback: int) -> pd.Series:
    """If USD has been rallying for N bars: short USD-quote pairs / long USD-base pairs."""
    usd_aligned = usd_idx.reindex(df.index).ffill()
    usd_change = usd_aligned.diff(lookback)
    s = pd.Series(0, index=df.index, dtype=int)
    if symbol in USD_QUOTE:
        s[usd_change > 0] = -1   # USD up → quote pair (e.g. EURUSD) down → SHORT
        s[usd_change < 0] = 1
    elif symbol in USD_BASE:
        s[usd_change > 0] = 1    # USD up → base pair (e.g. USDJPY) up → LONG
        s[usd_change < 0] = -1
    return s


def sig_usd_meanrev(df: pd.DataFrame, usd_idx: pd.Series, symbol: str, lookback: int, thr: float) -> pd.Series:
    """Fade USD extreme moves."""
    usd_aligned = usd_idx.reindex(df.index).ffill()
    z = zscore(usd_aligned, lookback)
    s = pd.Series(0, index=df.index, dtype=int)
    if symbol in USD_QUOTE:
        s[z > thr] = 1     # USD over-extended high → quote pair over-extended low → LONG
        s[z < -thr] = -1
    elif symbol in USD_BASE:
        s[z > thr] = -1
        s[z < -thr] = 1
    return s


# ---------- evaluation (non-overlapping) ----------

@dataclass
class EdgeResult:
    name: str; symbol: str; horizon: int
    n: int; gross: float; net: float; sharpe: float; t: float; hit: float

    def row(self): return (self.name, self.symbol, self.horizon, self.n,
                            round(self.gross, 3), round(self.net, 3),
                            round(self.sharpe, 3), round(self.t, 2),
                            round(self.hit * 100, 1))


def evaluate(df: pd.DataFrame, sig: pd.Series, horizon: int, symbol: str, name: str) -> EdgeResult | None:
    pip = PIP_SIZE[symbol]; cost = COSTS_PIPS[symbol]
    open_ = df["open"].values; s = sig.values; n = len(s)
    pnls = []
    i = 0
    while i < n - horizon - 1:
        if s[i] == 0:
            i += 1; continue
        e = open_[i + 1]; x = open_[i + 1 + horizon]
        if math.isnan(e) or math.isnan(x):
            i += 1; continue
        pnls.append((x - e) * s[i])
        i += horizon + 1
    if len(pnls) < 30:
        return None
    pips = np.array(pnls) / pip
    net = pips - cost
    m = float(net.mean()); sd = float(net.std(ddof=1))
    sh = m / sd if sd > 0 else 0.0
    t = m / (sd / math.sqrt(len(net))) if sd > 0 else 0.0
    return EdgeResult(name, symbol, horizon, len(net), float(pips.mean()), m, sh, t, float((net > 0).mean()))


def split(df: pd.DataFrame):
    mid = df.index[len(df) // 2]
    return df[df.index < mid], df[df.index >= mid]


# ---------- sweep ----------

PRICE_SIGNALS = [
    ("mom_24",     lambda d, _u, _s: sig_momentum(d, 24)),     # 1 day mom
    ("mom_120",    lambda d, _u, _s: sig_momentum(d, 120)),    # 5 day mom
    ("mom_240",    lambda d, _u, _s: sig_momentum(d, 240)),    # 10 day mom
    ("mr_z48_2",   lambda d, _u, _s: sig_meanrev_z(d, 48, 2.0)),
    ("mr_z120_2",  lambda d, _u, _s: sig_meanrev_z(d, 120, 2.0)),
    ("mr_z240_2.5",lambda d, _u, _s: sig_meanrev_z(d, 240, 2.5)),
    ("don_24",     lambda d, _u, _s: sig_donchian(d, 24)),
    ("don_120",    lambda d, _u, _s: sig_donchian(d, 120)),
    ("don_240",    lambda d, _u, _s: sig_donchian(d, 240)),
    ("don_fade_24",lambda d, _u, _s: -sig_donchian(d, 24)),
    ("dow_mon",    lambda d, _u, _s: sig_dow(d, {0}, set())),
    ("dow_fri",    lambda d, _u, _s: sig_dow(d, set(), {4})),
    ("dow_wed",    lambda d, _u, _s: sig_dow(d, {2}, set())),
    ("month_turn_long", lambda d, _u, _s: sig_month_turn(d, 2)),
]

USD_SIGNALS = [
    ("usd_mom_24",   lambda d, u, s: sig_usd_momentum(d, u, s, 24)),
    ("usd_mom_120",  lambda d, u, s: sig_usd_momentum(d, u, s, 120)),
    ("usd_mom_240",  lambda d, u, s: sig_usd_momentum(d, u, s, 240)),
    ("usd_mr_z120_2",lambda d, u, s: sig_usd_meanrev(d, u, s, 120, 2.0)),
    ("usd_mr_z240_2",lambda d, u, s: sig_usd_meanrev(d, u, s, 240, 2.0)),
]

HORIZONS = [4, 24, 72, 168]  # 4h, 1d, 3d, 1w


def run() -> None:
    data = load_h1()
    if not data:
        print("No data found.")
        return

    print("=== H1 DATA SUMMARY ===")
    for sym, df in sorted(data.items()):
        print(f"  {sym:8s} rows={len(df):>6d}  {df.index.min().date()}  ->  {df.index.max().date()}")

    usd = build_usd_index(data)
    print(f"\n=== USD STRENGTH INDEX: {len(usd)} aligned bars,",
          f"{usd.index.min().date() if len(usd) else 'NA'} -> {usd.index.max().date() if len(usd) else 'NA'} ===")

    rows_is, rows_oos = [], []

    for symbol, df in data.items():
        is_df, oos_df = split(df)
        # Price-only signals (all symbols)
        for name, fn in PRICE_SIGNALS:
            sig_is = fn(is_df, usd, symbol)
            sig_oos = fn(oos_df, usd, symbol)
            for h in HORIZONS:
                r_is = evaluate(is_df, sig_is, h, symbol, name)
                r_oos = evaluate(oos_df, sig_oos, h, symbol, name)
                if r_is: rows_is.append(r_is.row())
                if r_oos: rows_oos.append(r_oos.row())
        # USD-cross signals (USD pairs only)
        if symbol in USD_QUOTE | USD_BASE:
            for name, fn in USD_SIGNALS:
                sig_is = fn(is_df, usd, symbol)
                sig_oos = fn(oos_df, usd, symbol)
                for h in HORIZONS:
                    r_is = evaluate(is_df, sig_is, h, symbol, name)
                    r_oos = evaluate(oos_df, sig_oos, h, symbol, name)
                    if r_is: rows_is.append(r_is.row())
                    if r_oos: rows_oos.append(r_oos.row())

    cols = ["signal", "symbol", "h", "n", "gross", "net", "sharpe", "t", "hit%"]
    is_tbl = pd.DataFrame(rows_is, columns=cols)
    oos_tbl = pd.DataFrame(rows_oos, columns=cols)
    merged = is_tbl.merge(oos_tbl, on=["signal", "symbol", "h"], suffixes=("_is", "_oos"))

    profitable = merged[
        (merged["net_is"] > 0) & (merged["net_oos"] > 0)
        & (merged["t_is"] > 1.5) & (merged["t_oos"] > 1.0)
    ].sort_values("t_oos", ascending=False)

    print("\n=== PROFITABLE SIGNALS (net>0 BOTH halves, t_is>1.5, t_oos>1) ===")
    if profitable.empty:
        print("None.")
    else:
        print(profitable[["signal","symbol","h","n_is","net_is","t_is","hit%_is",
                          "n_oos","net_oos","t_oos","hit%_oos"]].to_string(index=False))

    print("\n=== TOP 15 BY t_oos (POSITIVE) ===")
    pos = merged[merged["net_oos"] > 0].sort_values("t_oos", ascending=False).head(15)
    print(pos[["signal","symbol","h","n_oos","net_oos","t_oos","hit%_oos","net_is","t_is"]].to_string(index=False))

    print("\n=== TOP 15 BY t_oos (NEGATIVE = invertible candidates) ===")
    neg = merged[merged["net_oos"] < 0].sort_values("t_oos").head(15)
    print(neg[["signal","symbol","h","n_oos","net_oos","t_oos","hit%_oos","net_is","t_is"]].to_string(index=False))


if __name__ == "__main__":
    run()
