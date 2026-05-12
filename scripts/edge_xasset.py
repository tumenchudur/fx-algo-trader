"""
Edge research with cross-asset signals on 2.8y of H1 data from yfinance.

Tests:
  - Price-only baselines (momentum, mean-rev, Donchian, Donchian-fade) on FX majors.
  - Cross-asset:
      VIX-spike risk-off (long JPY, short AUD/NZD)
      DXY momentum (FX should respond inversely / directly per pair)
      TNX (yield) momentum (USD should follow yields)
      SPX risk-on/off (correlation with AUD/NZD vs JPY/CHF)
      Gold-as-USD-inverse (XAUUSD up == USD weak)
  - Calendar: day-of-week, hour-of-day (tested separately by edge_hour_of_day_v2.py)
  - All trades NON-OVERLAPPING. Realistic round-trip costs.

Survivors are reported by t-stat consistency between the in-sample and out-of-sample
halves. We bias the criterion toward OOS (the half we did NOT pick from).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/yf")

COSTS_PIPS = {"EURUSD": 0.8, "GBPUSD": 1.0, "USDJPY": 1.0,
              "USDCHF": 1.5, "USDCAD": 1.2, "AUDUSD": 1.0, "NZDUSD": 1.5,
              "XAUUSD": 30.0}
PIP_SIZE = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2,
            "USDCHF": 1e-4, "USDCAD": 1e-4, "AUDUSD": 1e-4, "NZDUSD": 1e-4,
            "XAUUSD": 1e-2}
USD_QUOTE = {"EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "XAUUSD"}
USD_BASE = {"USDJPY", "USDCHF", "USDCAD"}
RISK_ON_PAIRS = {"AUDUSD", "NZDUSD"}        # rise on risk-on
RISK_OFF_PAIRS = {"USDJPY", "USDCHF"}       # USD-base; JPY/CHF strengthen on risk-off (=> pair falls)


def load(name: str) -> pd.DataFrame | None:
    p = DATA / f"{name}.parquet"
    if not p.exists():
        return None
    df = pd.read_parquet(p)
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def load_all() -> dict[str, pd.DataFrame]:
    out = {}
    for sym in ["EURUSD","GBPUSD","USDJPY","USDCHF","USDCAD","AUDUSD","NZDUSD","XAUUSD"]:
        d = load(sym)
        if d is not None:
            out[sym] = d[["open","high","low","close"]].astype(float)
    return out


def load_xasset() -> dict[str, pd.Series]:
    """Cross-asset close series."""
    out = {}
    for tk in ["DXY","TNX","VIX","SPX","NDX","WTI"]:
        d = load(tk)
        if d is not None and "close" in d.columns:
            out[tk] = d["close"].astype(float).rename(tk)
    return out


# ---------- indicators ----------

def zscore(s: pd.Series, n: int) -> pd.Series:
    return (s - s.rolling(n).mean()) / s.rolling(n).std()


# ---------- price signals ----------

def sig_mom(df: pd.DataFrame, n: int) -> pd.Series:
    r = df["close"].pct_change(n)
    s = pd.Series(0, index=df.index, dtype=int)
    s[r > 0] = 1; s[r < 0] = -1
    return s


def sig_mr_z(df: pd.DataFrame, n: int, thr: float) -> pd.Series:
    z = zscore(df["close"], n)
    s = pd.Series(0, index=df.index, dtype=int)
    s[z < -thr] = 1; s[z > thr] = -1
    return s


def sig_donchian(df: pd.DataFrame, n: int) -> pd.Series:
    up = df["high"].rolling(n).max().shift(1)
    dn = df["low"].rolling(n).min().shift(1)
    s = pd.Series(0, index=df.index, dtype=int)
    s[df["close"] > up] = 1; s[df["close"] < dn] = -1
    return s


# ---------- cross-asset signals ----------

def sig_vix_spike(df: pd.DataFrame, vix: pd.Series, sym: str, lookback: int, thr: float) -> pd.Series:
    """When VIX z-score > thr (risk-off): long JPY/CHF (= short USDJPY/USDCHF), short AUD/NZD."""
    z = zscore(vix.reindex(df.index, method="ffill"), lookback)
    s = pd.Series(0, index=df.index, dtype=int)
    if sym in RISK_ON_PAIRS:        # AUDUSD, NZDUSD: short on risk-off
        s[z > thr] = -1
        s[z < -thr] = 1
    elif sym in RISK_OFF_PAIRS:     # USDJPY, USDCHF: short on risk-off
        s[z > thr] = -1
        s[z < -thr] = 1
    return s


def sig_dxy_mom(df: pd.DataFrame, dxy: pd.Series, sym: str, lookback: int) -> pd.Series:
    """DXY momentum: trade FX pair in the direction implied by USD strength."""
    chg = dxy.reindex(df.index, method="ffill").diff(lookback)
    s = pd.Series(0, index=df.index, dtype=int)
    if sym in USD_QUOTE:
        s[chg > 0] = -1; s[chg < 0] = 1
    elif sym in USD_BASE:
        s[chg > 0] = 1; s[chg < 0] = -1
    return s


def sig_tnx_mom(df: pd.DataFrame, tnx: pd.Series, sym: str, lookback: int) -> pd.Series:
    """Higher US yields → stronger USD."""
    chg = tnx.reindex(df.index, method="ffill").diff(lookback)
    s = pd.Series(0, index=df.index, dtype=int)
    if sym in USD_QUOTE:
        s[chg > 0] = -1; s[chg < 0] = 1
    elif sym in USD_BASE:
        s[chg > 0] = 1; s[chg < 0] = -1
    return s


def sig_spx_risk(df: pd.DataFrame, spx: pd.Series, sym: str, lookback: int) -> pd.Series:
    """SPX rising = risk-on; AUD/NZD up, JPY/CHF up vs USD inverted."""
    chg = spx.reindex(df.index, method="ffill").pct_change(lookback)
    s = pd.Series(0, index=df.index, dtype=int)
    if sym in RISK_ON_PAIRS:
        s[chg > 0] = 1; s[chg < 0] = -1
    elif sym in RISK_OFF_PAIRS:
        s[chg > 0] = 1; s[chg < 0] = -1   # USDJPY, USDCHF rise when SPX rises
    return s


def sig_dow(df: pd.DataFrame, long_dows: set[int], short_dows: set[int]) -> pd.Series:
    dow = df.index.dayofweek; h = df.index.hour
    first = h == 0
    s = pd.Series(0, index=df.index, dtype=int)
    s[first & np.isin(dow, list(long_dows))] = 1
    s[first & np.isin(dow, list(short_dows))] = -1
    return s


# ---------- evaluation ----------

@dataclass
class R:
    name: str; sym: str; h: int; n: int; gross: float; net: float; sh: float; t: float; hit: float
    def row(self): return (self.name, self.sym, self.h, self.n,
                            round(self.gross,3), round(self.net,3),
                            round(self.sh,3), round(self.t,2), round(self.hit*100,1))


def evaluate(df: pd.DataFrame, sig: pd.Series, h: int, sym: str, name: str) -> R | None:
    pip = PIP_SIZE[sym]; cost = COSTS_PIPS[sym]
    open_ = df["open"].values; s = sig.values
    n = len(s); pnls = []
    i = 0
    while i < n - h - 1:
        if s[i] == 0:
            i += 1; continue
        e = open_[i+1]; x = open_[i+1+h]
        if math.isnan(e) or math.isnan(x):
            i += 1; continue
        pnls.append((x - e) * s[i])
        i += h + 1
    if len(pnls) < 30:
        return None
    pips = np.array(pnls) / pip
    net = pips - cost
    m = float(net.mean()); sd = float(net.std(ddof=1))
    sh = m / sd if sd > 0 else 0.0
    t = m / (sd / math.sqrt(len(net))) if sd > 0 else 0.0
    return R(name, sym, h, len(net), float(pips.mean()), m, sh, t, float((net > 0).mean()))


def split(df: pd.DataFrame):
    mid = df.index[len(df) // 2]
    return df[df.index < mid], df[df.index >= mid]


HORIZONS = [4, 24, 72, 168]   # 4h, 1d, 3d, 1wk


def build_signals(symbol: str):
    """Return list of (name, builder(df) -> Series)."""
    sigs = [
        ("mom_24",      lambda d: sig_mom(d, 24)),
        ("mom_72",      lambda d: sig_mom(d, 72)),
        ("mom_240",     lambda d: sig_mom(d, 240)),
        ("mr_z48_2",    lambda d: sig_mr_z(d, 48, 2.0)),
        ("mr_z168_2",   lambda d: sig_mr_z(d, 168, 2.0)),
        ("don_24",      lambda d: sig_donchian(d, 24)),
        ("don_120",     lambda d: sig_donchian(d, 120)),
        ("don_fade_24", lambda d: -sig_donchian(d, 24)),
        ("dow_mon_long",lambda d: sig_dow(d, {0}, set())),
        ("dow_fri_short",lambda d: sig_dow(d, set(), {4})),
        ("dow_thu_long",lambda d: sig_dow(d, {3}, set())),
    ]
    return sigs


def build_xasset_signals(symbol: str, x: dict[str, pd.Series]):
    sigs = []
    if "VIX" in x and symbol in (RISK_ON_PAIRS | RISK_OFF_PAIRS):
        sigs += [
            ("vix_spike_z48_1.5",  lambda d, x=x: sig_vix_spike(d, x["VIX"], symbol, 48, 1.5)),
            ("vix_spike_z120_2",   lambda d, x=x: sig_vix_spike(d, x["VIX"], symbol, 120, 2.0)),
        ]
    if "DXY" in x and symbol in (USD_QUOTE | USD_BASE):
        sigs += [
            ("dxy_mom_24",  lambda d, x=x: sig_dxy_mom(d, x["DXY"], symbol, 24)),
            ("dxy_mom_120", lambda d, x=x: sig_dxy_mom(d, x["DXY"], symbol, 120)),
        ]
    if "TNX" in x and symbol in (USD_QUOTE | USD_BASE):
        sigs += [
            ("tnx_mom_24",  lambda d, x=x: sig_tnx_mom(d, x["TNX"], symbol, 24)),
            ("tnx_mom_120", lambda d, x=x: sig_tnx_mom(d, x["TNX"], symbol, 120)),
        ]
    if "SPX" in x and symbol in (RISK_ON_PAIRS | RISK_OFF_PAIRS):
        sigs += [
            ("spx_mom_24",  lambda d, x=x: sig_spx_risk(d, x["SPX"], symbol, 24)),
            ("spx_mom_120", lambda d, x=x: sig_spx_risk(d, x["SPX"], symbol, 120)),
        ]
    return sigs


def run() -> None:
    data = load_all()
    xa = load_xasset()
    print("=== H1 PAIRS ===")
    for s, d in sorted(data.items()):
        print(f"  {s:8s} rows={len(d):>6d}  {d.index.min().date()} -> {d.index.max().date()}")
    print("=== CROSS-ASSET ===")
    for k, s in sorted(xa.items()):
        print(f"  {k:8s} rows={len(s):>6d}  {s.index.min().date()} -> {s.index.max().date()}")

    rows_is, rows_oos = [], []
    for sym, df in data.items():
        is_df, oos_df = split(df)
        sigs = build_signals(sym) + build_xasset_signals(sym, xa)
        for name, fn in sigs:
            sis = fn(is_df); soos = fn(oos_df)
            for h in HORIZONS:
                r1 = evaluate(is_df, sis, h, sym, name)
                r2 = evaluate(oos_df, soos, h, sym, name)
                if r1: rows_is.append(r1.row())
                if r2: rows_oos.append(r2.row())

    cols = ["signal","symbol","h","n","gross","net","sh","t","hit%"]
    is_t = pd.DataFrame(rows_is, columns=cols)
    oos_t = pd.DataFrame(rows_oos, columns=cols)
    m = is_t.merge(oos_t, on=["signal","symbol","h"], suffixes=("_is","_oos"))

    pos_both = m[(m.net_is > 0) & (m.net_oos > 0)]
    print(f"\n=== {len(pos_both)} signals POSITIVE in BOTH halves (out of {len(m)}) ===")
    print(pos_both.sort_values("t_oos", ascending=False).head(25)[
        ["signal","symbol","h","n_is","net_is","t_is","n_oos","net_oos","t_oos","hit%_oos"]
    ].to_string(index=False))

    survivors = m[(m.net_is > 0) & (m.net_oos > 0) & (m.t_is > 1.5) & (m.t_oos > 1.5)]
    print(f"\n=== ROBUST SURVIVORS (net>0 BOTH, t>1.5 BOTH) — {len(survivors)} found ===")
    if survivors.empty:
        print("None.")
    else:
        print(survivors.sort_values("t_oos", ascending=False)[
            ["signal","symbol","h","n_is","net_is","t_is","hit%_is",
             "n_oos","net_oos","t_oos","hit%_oos"]
        ].to_string(index=False))

    # Inverted signals (consistently negative)
    inv = m[(m.net_is < 0) & (m.net_oos < 0) & (m.t_is < -1.5) & (m.t_oos < -1.5)]
    print(f"\n=== INVERTIBLE (consistently negative; flip sign for edge) — {len(inv)} ===")
    if not inv.empty:
        print(inv.sort_values("t_oos")[
            ["signal","symbol","h","n_is","net_is","t_is","n_oos","net_oos","t_oos"]
        ].to_string(index=False))


if __name__ == "__main__":
    run()
