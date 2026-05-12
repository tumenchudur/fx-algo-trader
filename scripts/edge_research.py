"""
Edge research: test raw signals on real M5/H1 data.

Methodology:
- Generate signals (no SL/TP, no filters).
- For each signal, compute forward-N-bar return (entry at next-bar open, exit at open after N bars).
- Apply realistic round-trip cost: spread + slippage.
- Report: n trades, mean return (pips), Sharpe (per-trade), t-stat vs zero, hit rate.
- Split in-sample / out-of-sample by time. An edge that doesn't survive OOS is noise.

A signal is interesting if BOTH halves show t-stat > ~2 with same sign after costs.
"""

from __future__ import annotations

import glob
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

# ---------- cost assumptions (round-trip, in price units) ----------
# Realistic Exness Pro / IC Markets spreads on liquid hours, plus 0.2pip slippage.
COSTS_PIPS = {
    "EURUSD": 0.8,
    "GBPUSD": 1.0,
    "USDJPY": 1.0,
    "XAUUSD": 25.0,  # ~$0.25 per oz round-trip; XAU pip = $0.01
}
PIP_SIZE = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "XAUUSD": 1e-2}


@dataclass
class EdgeResult:
    name: str
    symbol: str
    horizon_bars: int
    n_trades: int
    mean_pips_gross: float
    mean_pips_net: float
    sharpe: float
    t_stat: float
    hit_rate: float

    def row(self) -> tuple:
        return (
            self.name,
            self.symbol,
            self.horizon_bars,
            self.n_trades,
            round(self.mean_pips_gross, 3),
            round(self.mean_pips_net, 3),
            round(self.sharpe, 3),
            round(self.t_stat, 2),
            round(self.hit_rate * 100, 1),
        )


def load_all() -> dict[str, pd.DataFrame]:
    """Load every available real-data parquet, prefer longest series per symbol."""
    bag: dict[str, list[pd.DataFrame]] = {}
    for f in sorted(
        glob.glob("data/sampl/*_M5.parquet")
        + glob.glob("data/sample/*_M5.parquet")
        + glob.glob("data/processed/*_M5.parquet")
    ):
        sym = Path(f).stem.split("_")[0]
        df = pd.read_parquet(f)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        bag.setdefault(sym, []).append(df)

    out: dict[str, pd.DataFrame] = {}
    for sym, frames in bag.items():
        merged = pd.concat(frames).sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]
        # Keep only standard cols
        for c in ["open", "high", "low", "close"]:
            assert c in merged.columns, f"{sym} missing {c}"
        out[sym] = merged[["open", "high", "low", "close"]].astype(float)
    return out


# ---------- indicators (vectorized, no lookahead — using only past data) ----------

def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(period).mean()
    loss = (-delta.clip(upper=0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - 100 / (1 + rs)


def atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    tr = pd.concat(
        [(high - low).abs(), (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    return tr.rolling(period).mean()


def zscore(close: pd.Series, period: int = 50) -> pd.Series:
    mean = close.rolling(period).mean()
    std = close.rolling(period).std()
    return (close - mean) / std


# ---------- signal generators ----------
# Each returns a Series of {-1, 0, +1} indexed like close. The signal at time t
# is acted on at time t+1 (entry = open[t+1]) and exited at open[t+1+horizon].

def sig_momentum(df: pd.DataFrame, lookback: int) -> pd.Series:
    ret = df["close"].pct_change(lookback)
    s = pd.Series(0, index=df.index, dtype=int)
    s[ret > 0] = 1
    s[ret < 0] = -1
    return s


def sig_meanrev_z(df: pd.DataFrame, lookback: int, threshold: float) -> pd.Series:
    z = zscore(df["close"], lookback)
    s = pd.Series(0, index=df.index, dtype=int)
    s[z < -threshold] = 1   # oversold → long
    s[z > threshold] = -1   # overbought → short
    return s


def sig_rsi_extremes(df: pd.DataFrame, period: int, lo: float, hi: float) -> pd.Series:
    r = rsi(df["close"], period)
    s = pd.Series(0, index=df.index, dtype=int)
    s[r < lo] = 1
    s[r > hi] = -1
    return s


def sig_breakout(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Donchian breakout — the strategy in the repo, raw."""
    upper = df["high"].rolling(lookback).max().shift(1)
    lower = df["low"].rolling(lookback).min().shift(1)
    s = pd.Series(0, index=df.index, dtype=int)
    s[df["close"] > upper] = 1
    s[df["close"] < lower] = -1
    return s


def sig_breakout_fade(df: pd.DataFrame, lookback: int) -> pd.Series:
    """Inverse of donchian breakout — fade it."""
    return -sig_breakout(df, lookback)


def sig_hour_of_day(df: pd.DataFrame, long_hours: set[int], short_hours: set[int]) -> pd.Series:
    """Fire only at the FIRST bar of each listed hour (minute == 0) so trades don't overlap."""
    h = df.index.hour
    m = df.index.minute
    first = m == 0
    s = pd.Series(0, index=df.index, dtype=int)
    s[first & np.isin(h, list(long_hours))] = 1
    s[first & np.isin(h, list(short_hours))] = -1
    return s


def sig_overnight(df: pd.DataFrame, asia_hours_utc: range) -> pd.Series:
    """Long at Asia close (08:00 UTC), exit at NY close. Tests session bias."""
    h = df.index.hour
    m = df.index.minute
    s = pd.Series(0, index=df.index, dtype=int)
    s[(h == 8) & (m == 0)] = 1
    return s


# ---------- backtest a single signal ----------

def evaluate(
    df: pd.DataFrame,
    signal: pd.Series,
    horizon: int,
    symbol: str,
    name: str,
) -> EdgeResult | None:
    """
    Forward-N-bar return per trade, net of costs.

    Trades are NON-OVERLAPPING: once a trade fires at bar t, the next eligible
    bar is t + horizon. This makes the per-trade samples independent so the
    t-stat is not inflated by overlap.
    """
    pip = PIP_SIZE[symbol]
    cost_pips = COSTS_PIPS[symbol]
    open_ = df["open"].values
    sig = signal.values
    n = len(sig)

    pnls: list[float] = []
    i = 0
    while i < n - horizon - 1:
        s = sig[i]
        if s == 0:
            i += 1
            continue
        entry = open_[i + 1]
        exit_ = open_[i + 1 + horizon]
        if math.isnan(entry) or math.isnan(exit_):
            i += 1
            continue
        pnls.append((exit_ - entry) * s)
        i += horizon + 1  # skip past the trade window

    if len(pnls) < 30:
        return None

    pips = np.array(pnls) / pip
    pips_net = pips - cost_pips  # cost on every trade

    mean = float(pips_net.mean())
    std = float(pips_net.std(ddof=1))
    sharpe = mean / std if std > 0 else 0.0
    t = mean / (std / math.sqrt(len(pips_net))) if std > 0 else 0.0
    hit = float((pips_net > 0).mean())

    return EdgeResult(
        name=name,
        symbol=symbol,
        horizon_bars=horizon,
        n_trades=len(pips_net),
        mean_pips_gross=float(pips.mean()),
        mean_pips_net=mean,
        sharpe=sharpe,
        t_stat=t,
        hit_rate=hit,
    )


def buy_and_hold_pips(df: pd.DataFrame, symbol: str, horizon: int) -> float:
    """Mean per-trade pip move of just being long, non-overlapping, gross of costs."""
    pip = PIP_SIZE[symbol]
    open_ = df["open"].values
    n = len(open_)
    moves = []
    i = 0
    while i < n - horizon - 1:
        moves.append(open_[i + 1 + horizon] - open_[i + 1])
        i += horizon + 1
    return float(np.mean(moves) / pip) if moves else 0.0


# ---------- main sweep ----------

SIGNALS: list[tuple[str, callable]] = [
    ("momentum_5",         lambda d: sig_momentum(d, 5)),
    ("momentum_20",        lambda d: sig_momentum(d, 20)),
    ("momentum_50",        lambda d: sig_momentum(d, 50)),
    ("meanrev_z20_2",      lambda d: sig_meanrev_z(d, 20, 2.0)),
    ("meanrev_z50_2",      lambda d: sig_meanrev_z(d, 50, 2.0)),
    ("meanrev_z100_2.5",   lambda d: sig_meanrev_z(d, 100, 2.5)),
    ("rsi14_30_70",        lambda d: sig_rsi_extremes(d, 14, 30, 70)),
    ("rsi14_20_80",        lambda d: sig_rsi_extremes(d, 14, 20, 80)),
    ("rsi5_20_80",         lambda d: sig_rsi_extremes(d, 5, 20, 80)),
    ("donchian_20",        lambda d: sig_breakout(d, 20)),
    ("donchian_50",        lambda d: sig_breakout(d, 50)),
    ("donchian_100",       lambda d: sig_breakout(d, 100)),
    ("donchian_fade_20",   lambda d: sig_breakout_fade(d, 20)),
    ("donchian_fade_50",   lambda d: sig_breakout_fade(d, 50)),
    # Session-based — ranges roughly: Asia 0-7, London 7-12, NY 12-21
    ("london_long",        lambda d: sig_hour_of_day(d, {7, 8, 9}, set())),
    ("london_short",       lambda d: sig_hour_of_day(d, set(), {7, 8, 9})),
    ("ny_long",            lambda d: sig_hour_of_day(d, {13, 14, 15}, set())),
    ("ny_short",           lambda d: sig_hour_of_day(d, set(), {13, 14, 15})),
    ("asia_fade",          lambda d: sig_hour_of_day(d, set(), {1, 2, 3, 4, 5})),
]

HORIZONS = [12, 48, 288]  # M5: 1h, 4h, 1day


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mid = df.index[len(df) // 2]
    return df[df.index < mid], df[df.index >= mid]


def run() -> None:
    data = load_all()
    rows_is, rows_oos = [], []

    print("\n=== DATA & BUY-AND-HOLD DRIFT (gross pips per N-bar window) ===")
    print(f"{'symbol':10s} {'rows':>7s} {'span':>40s}  bnh_h12  bnh_h48  bnh_h288")
    for symbol, df in sorted(data.items()):
        bnh = {h: buy_and_hold_pips(df, symbol, h) for h in HORIZONS}
        span = f"{df.index.min().date()} -> {df.index.max().date()}"
        print(f"{symbol:10s} {len(df):>7d} {span:>40s}  {bnh[12]:7.2f}  {bnh[48]:7.2f}  {bnh[288]:7.2f}")

    for symbol, df in data.items():
        is_df, oos_df = split(df)
        for name, fn in SIGNALS:
            sig_is = fn(is_df)
            sig_oos = fn(oos_df)
            for h in HORIZONS:
                r_is = evaluate(is_df, sig_is, h, symbol, name)
                r_oos = evaluate(oos_df, sig_oos, h, symbol, name)
                if r_is:
                    rows_is.append(r_is.row())
                if r_oos:
                    rows_oos.append(r_oos.row())

    cols = ["signal", "symbol", "h", "n", "gross_pips", "net_pips", "sharpe", "t_stat", "hit%"]
    is_tbl = pd.DataFrame(rows_is, columns=cols)
    oos_tbl = pd.DataFrame(rows_oos, columns=cols)

    merged = is_tbl.merge(oos_tbl, on=["signal", "symbol", "h"], suffixes=("_is", "_oos"))

    # Profitable in BOTH halves (positive net_pips), |t_stat_is|>2, |t_stat_oos|>1
    profitable = merged[
        (merged["net_pips_is"] > 0)
        & (merged["net_pips_oos"] > 0)
        & (merged["t_stat_is"] > 2.0)
        & (merged["t_stat_oos"] > 1.0)
    ].sort_values("t_stat_oos", ascending=False)

    # Anti-edges: consistently NEGATIVE → invertible
    inverted = merged[
        (merged["net_pips_is"] < 0)
        & (merged["net_pips_oos"] < 0)
        & (merged["t_stat_is"] < -2.0)
        & (merged["t_stat_oos"] < -1.0)
    ].sort_values("t_stat_oos")

    print("\n=== PROFITABLE SIGNALS (positive net pips, IS t>2, OOS t>1, BOTH halves) ===")
    if len(profitable) == 0:
        print("None.")
    else:
        print(profitable[
            ["signal", "symbol", "h",
             "n_is", "net_pips_is", "t_stat_is", "hit%_is",
             "n_oos", "net_pips_oos", "t_stat_oos", "hit%_oos"]
        ].to_string(index=False))

    print("\n=== ANTI-EDGES (consistently lose; consider INVERTING the rule) ===")
    if len(inverted) == 0:
        print("None.")
    else:
        print(inverted[
            ["signal", "symbol", "h",
             "n_is", "net_pips_is", "t_stat_is",
             "n_oos", "net_pips_oos", "t_stat_oos"]
        ].head(15).to_string(index=False))


if __name__ == "__main__":
    run()
