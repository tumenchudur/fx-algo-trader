"""
Verify the survivor signals from edge_xasset.py.

Checks:
1. Buy-and-hold drift of XAUUSD over the sample — does dow_mon_long beat random Monday?
2. Equity curves for the 5 survivors. Edge should be diffuse, not from 1-2 outliers.
3. Combined "yield-strength" basket: long USDJPY + short AUDUSD when TNX rises.
4. Inverse short-term momentum on AUDUSD/NZDUSD/USDCHF/USDCAD (the anti-edges flipped).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path("data/yf")
PIP = {"EURUSD":1e-4,"GBPUSD":1e-4,"USDJPY":1e-2,"USDCHF":1e-4,"USDCAD":1e-4,
       "AUDUSD":1e-4,"NZDUSD":1e-4,"XAUUSD":1e-2}
COST = {"EURUSD":0.8,"GBPUSD":1.0,"USDJPY":1.0,"USDCHF":1.5,"USDCAD":1.2,
        "AUDUSD":1.0,"NZDUSD":1.5,"XAUUSD":30.0}


def load(name: str) -> pd.DataFrame:
    df = pd.read_parquet(DATA / f"{name}.parquet")
    df = df[~df.index.duplicated(keep="first")].sort_index()
    if df.index.tz is None:
        df.index = df.index.tz_localize("UTC")
    return df


def ttest_mean(x: np.ndarray) -> tuple[float, float]:
    if len(x) < 2: return 0.0, 0.0
    sd = x.std(ddof=1)
    if sd == 0: return float(x.mean()), 0.0
    return float(x.mean()), float(x.mean() / (sd / math.sqrt(len(x))))


# ---------- 1. Gold drift sanity check ----------

def check_gold_monday(df: pd.DataFrame, horizon: int) -> None:
    """Compare Monday-open long return vs same-horizon random-day long returns."""
    pip = PIP["XAUUSD"]; cost = COST["XAUUSD"]
    open_ = df["open"]
    # All possible non-overlapping H1 entries: every `horizon+1` bars
    bars = open_.values
    times = open_.index

    # Monday entries: dow==0, hour==0
    monday_mask = (times.dayofweek == 0) & (times.hour == 0)

    # Forward returns from each bar i to i+horizon (in pips, gross)
    fwd = (bars[horizon:] - bars[:-horizon])[:-1] / pip
    fwd_idx = times[:-horizon-1]

    monday_idx = monday_mask[:-horizon-1]
    other_idx = ~monday_idx

    mon_ret = fwd[monday_idx]
    other_ret = fwd[other_idx]
    mon_net = mon_ret - cost
    other_net = other_ret - cost

    print(f"\n=== XAUUSD long, horizon={horizon}h: Monday vs other days ===")
    print(f"  Monday-open  n={len(mon_net):>4d}  mean_net={mon_net.mean():>10.2f} pips  t={ttest_mean(mon_net)[1]:>5.2f}  hit%={(mon_net>0).mean()*100:.1f}")
    print(f"  Other-bars   n={len(other_net):>4d}  mean_net={other_net.mean():>10.2f} pips  t={ttest_mean(other_net)[1]:>5.2f}  hit%={(other_net>0).mean()*100:.1f}")
    print(f"  Difference   = {mon_net.mean() - other_net.mean():+.2f} pips/trade")
    # Welch t-test for the *difference*
    n1, n2 = len(mon_net), len(other_net)
    s1, s2 = mon_net.std(ddof=1), other_net.std(ddof=1)
    se = math.sqrt(s1*s1/n1 + s2*s2/n2)
    t_diff = (mon_net.mean() - other_net.mean()) / se if se > 0 else 0
    print(f"  Welch t (Mon vs other)   = {t_diff:.2f}   <-- the actual edge test")


# ---------- 2. Yield-driven basket ----------

def yield_basket(usdjpy: pd.DataFrame, audusd: pd.DataFrame, tnx: pd.Series, lookback: int, horizon: int) -> None:
    """Combined: long USDJPY + short AUDUSD when TNX rises over `lookback` hours.
    Per-trade PnL in normalized vol-adjusted units."""
    print(f"\n=== Yield basket (long USDJPY + short AUDUSD on TNX rise), lookback={lookback}h horizon={horizon}h ===")
    # Align all on USDJPY index
    idx = usdjpy.index
    audusd = audusd.reindex(idx, method="ffill")
    tnx = tnx.reindex(idx, method="ffill")

    chg = tnx.diff(lookback)
    open_jpy = usdjpy["open"].values
    open_aud = audusd["open"].values
    sig = chg.values

    # Normalize each leg by its sample stdev so they contribute equally
    jpy_returns_full = pd.Series(open_jpy).pct_change(horizon).dropna()
    aud_returns_full = pd.Series(open_aud).pct_change(horizon).dropna()
    jpy_std = jpy_returns_full.std()
    aud_std = aud_returns_full.std()

    pnls = []
    n = len(sig)
    i = lookback
    while i < n - horizon - 1:
        s = sig[i]
        if math.isnan(s) or s == 0:
            i += 1; continue
        direction = 1 if s > 0 else -1   # TNX up: long USD strength
        e_jpy = open_jpy[i+1]; x_jpy = open_jpy[i+1+horizon]
        e_aud = open_aud[i+1]; x_aud = open_aud[i+1+horizon]
        if any(math.isnan(v) for v in (e_jpy, x_jpy, e_aud, x_aud)):
            i += 1; continue
        # Long USDJPY (USD up) + short AUDUSD (USD up)
        ret_jpy = (x_jpy - e_jpy) / e_jpy * direction
        ret_aud = -(x_aud - e_aud) / e_aud * direction
        # Vol-normalize
        ret_jpy /= jpy_std
        ret_aud /= aud_std
        pnls.append(0.5 * (ret_jpy + ret_aud))
        i += horizon + 1

    pnls = np.array(pnls)
    if len(pnls) < 30:
        print("  Too few trades.")
        return
    m, t = ttest_mean(pnls)
    sh = m / pnls.std(ddof=1) if pnls.std(ddof=1) > 0 else 0
    print(f"  n={len(pnls)}  mean_normret={m:.4f}  t={t:.2f}  sharpe(per trade)={sh:.3f}  hit%={(pnls>0).mean()*100:.1f}")
    # Rough: with 4-week-equivalent rate of trades and Sharpe...
    trades_per_year = len(pnls) * (8760 / horizon) / (len(idx))   # approx
    print(f"  Approx trades/yr={trades_per_year:.0f}  →  approx annualized Sharpe≈{sh * math.sqrt(trades_per_year):.2f}")


# ---------- 3. Equity curves of survivors ----------

def equity_curve(df: pd.DataFrame, sig: np.ndarray, horizon: int, sym: str, name: str) -> None:
    pip = PIP[sym]; cost = COST[sym]
    open_ = df["open"].values; n = len(sig)
    pnls = []; times = []
    i = 0
    while i < n - horizon - 1:
        if sig[i] == 0:
            i += 1; continue
        e = open_[i+1]; x = open_[i+1+horizon]
        if math.isnan(e) or math.isnan(x):
            i += 1; continue
        pnls.append((x - e) * sig[i] / pip - cost)
        times.append(df.index[i+1+horizon])
        i += horizon + 1
    pnls = np.array(pnls)
    cum = pnls.cumsum()
    if len(pnls) < 10: return
    # Split into halves and quarters to see where the edge lives
    quarters = np.array_split(pnls, 4)
    print(f"  {name:24s} {sym:7s} h={horizon:3d}  total_pnl={cum[-1]:>10.1f}  n={len(pnls):>4d}")
    for j, q in enumerate(quarters):
        m, t = ttest_mean(q)
        print(f"    quarter {j+1}: n={len(q):>3d}  mean={m:>8.2f}  t={t:>5.2f}  cumPnL={q.sum():>10.1f}")


# ---------- 4. Inverse short-term momentum (mean reversion at h=4) ----------

def test_mr_4h(symbols: list[str], lookback: int = 24) -> None:
    """Anti-edges said: 24h momentum on AUDUSD/NZDUSD/USDCHF/USDCAD at h=4 LOSES.
    So short-term mean reversion may have edge. Test the inverse signal directly."""
    print(f"\n=== Short-term mean reversion (invert mom_{lookback}, hold 4h) on risk pairs ===")
    for sym in symbols:
        df = load(sym)
        df = df[["open","high","low","close"]].astype(float)
        # Split halves
        mid = df.index[len(df)//2]
        for label, sub in [("IS", df[df.index < mid]), ("OOS", df[df.index >= mid])]:
            r = sub["close"].pct_change(lookback)
            sig = pd.Series(0, index=sub.index, dtype=int)
            sig[r > 0] = -1   # invert
            sig[r < 0] = 1
            pip = PIP[sym]; cost = COST[sym]
            open_ = sub["open"].values; s = sig.values
            n = len(s); pnls = []
            i = 0
            while i < n - 4 - 1:
                if s[i] == 0:
                    i += 1; continue
                e = open_[i+1]; x = open_[i+5]
                if math.isnan(e) or math.isnan(x):
                    i += 1; continue
                pnls.append((x - e) * s[i] / pip - cost)
                i += 5
            pnls = np.array(pnls)
            if len(pnls) > 30:
                m, t = ttest_mean(pnls)
                hit = (pnls > 0).mean() * 100
                print(f"  {sym:7s} {label:3s}  n={len(pnls):>4d}  mean_net={m:>6.2f} pips  t={t:>5.2f}  hit%={hit:.1f}")


def main() -> None:
    # 1. Gold sanity
    xau = load("XAUUSD"); xau = xau[["open","high","low","close"]]
    for h in [24, 72, 168]:
        check_gold_monday(xau, h)

    # 2. Yield basket
    usdjpy = load("USDJPY"); audusd = load("AUDUSD"); tnx = load("TNX")["close"]
    for lb, h in [(24, 72), (120, 168)]:
        yield_basket(usdjpy, audusd, tnx, lb, h)

    # 3. Equity curves for survivors
    print("\n=== EDGE TIMING (per quarter PnL split) ===")
    # Yield-mom USDJPY h=72 (lookback 24)
    df = load("USDJPY"); df = df[["open","high","low","close"]]
    tnx_a = tnx.reindex(df.index, method="ffill")
    sig = np.zeros(len(df), dtype=int)
    chg = tnx_a.diff(24).values
    sig[chg > 0] = 1; sig[chg < 0] = -1
    equity_curve(df, sig, 72, "USDJPY", "tnx_mom_24")

    # Yield-mom AUDUSD h=168 (lookback 120)
    df = load("AUDUSD"); df = df[["open","high","low","close"]]
    tnx_a = tnx.reindex(df.index, method="ffill")
    sig = np.zeros(len(df), dtype=int)
    chg = tnx_a.diff(120).values
    sig[chg > 0] = -1; sig[chg < 0] = 1
    equity_curve(df, sig, 168, "AUDUSD", "tnx_mom_120")

    # XAUUSD Monday long h=72
    df = load("XAUUSD"); df = df[["open","high","low","close"]]
    sig = np.zeros(len(df), dtype=int)
    mask = (df.index.dayofweek == 0) & (df.index.hour == 0)
    sig[mask] = 1
    equity_curve(df, sig, 72, "XAUUSD", "dow_mon_long")
    equity_curve(df, sig, 168, "XAUUSD", "dow_mon_long")

    # 4. Mean reversion 4h on risk pairs
    test_mr_4h(["AUDUSD", "NZDUSD", "USDCHF", "USDCAD"], lookback=24)


if __name__ == "__main__":
    main()
