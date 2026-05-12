"""
Carry + 12-month momentum combined factor (Asness/Moskowitz "Value & Momentum
Everywhere", 2013).

Hypothesis: combining carry rank with 12m-momentum rank produces a higher
Sharpe than carry alone, because momentum partly hedges the carry-unwind risk.

Three variants compared:
  - carry-only (baseline)
  - mom-only (12m total return rank)
  - 50/50 carry+momentum (rank-average then long top-K / short bottom-K)
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FX = Path("data/yf_d1")
DATA_RATES = Path("data/rates")

CCY_PAIR = {
    "EUR": ("EURUSD", False), "GBP": ("GBPUSD", False),
    "AUD": ("AUDUSD", False), "NZD": ("NZDUSD", False),
    "JPY": ("USDJPY", True),  "CHF": ("USDCHF", True),
    "CAD": ("USDCAD", True),
}


def load_fx() -> pd.DataFrame:
    cols = {}
    for ccy, (pair, invert) in CCY_PAIR.items():
        df = pd.read_parquet(DATA_FX / f"{pair}.parquet")
        df = df[~df.index.duplicated(keep="first")].sort_index()
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        c = df["close"].astype(float)
        if invert:
            c = 1.0 / c
        cols[ccy] = c.rename(ccy)
    return pd.concat(cols.values(), axis=1).sort_index()


def load_rates() -> pd.DataFrame:
    return pd.read_parquet(DATA_RATES / "short_rates.parquet")


def make_signal_from_rank(score: pd.DataFrame, k: int) -> pd.DataFrame:
    """Cross-sectional: long top-K, short bottom-K each row."""
    sig = pd.DataFrame(0.0, index=score.index, columns=score.columns)
    for d, row in score.iterrows():
        ranked = row.dropna().rank(method="first")
        n = len(ranked)
        if n < 2 * k:
            continue
        sig.loc[d, ranked[ranked > n - k].index] = 1
        sig.loc[d, ranked[ranked <= k].index] = -1
    return sig


def backtest(score: pd.DataFrame, fx: pd.DataFrame, rates: pd.DataFrame,
             k: int = 2, vol_window: int = 60, target_vol: float = 0.10,
             label: str = "") -> pd.Series:
    """Common backtest plumbing given a `score` DataFrame (currency rank input)."""
    log_ret = np.log(fx / fx.shift(1))
    carry_daily = (rates.drop(columns="USD").subtract(rates["USD"], axis=0)) / 100.0 / 252.0
    realized_vol = log_ret.rolling(vol_window).std() * math.sqrt(252)

    raw_sig = make_signal_from_rank(score, k)

    is_rebal = fx.index.to_series().groupby(fx.index.to_period("M")).transform(
        lambda s: s == s.iloc[0]
    ).values

    weights = pd.DataFrame(0.0, index=fx.index, columns=raw_sig.columns)
    current_w = pd.Series(0.0, index=raw_sig.columns)
    for i, d in enumerate(fx.index):
        if is_rebal[i] and d in raw_sig.index:
            s = raw_sig.loc[d]
            v = realized_vol.iloc[i]
            if not v.isna().any():
                leg_w = (s / v.replace(0, np.nan)).fillna(0)
                gross = leg_w.abs().sum()
                if gross > 0:
                    leg_w = leg_w / gross
                    bv = float((leg_w.abs() * v).pow(2).sum() ** 0.5)
                    if bv > 0:
                        leg_w = leg_w * (target_vol / bv)
                    current_w = leg_w
        weights.iloc[i] = current_w.values

    spot = (weights.shift(1) * log_ret).sum(axis=1).fillna(0)
    carry = (weights.shift(1) * carry_daily).sum(axis=1).fillna(0)
    cost = (weights.diff().abs().fillna(0) * 0.0001).sum(axis=1).fillna(0)
    return (spot + carry - cost).rename(label)


def metrics(pnl: pd.Series, label: str) -> dict:
    pnl = pnl.dropna()
    pnl = pnl.loc[pnl.ne(0).idxmax():]   # drop leading zeros
    if len(pnl) < 30:
        return {"label": label, "n": len(pnl)}
    mu = pnl.mean() * 252
    sd = pnl.std(ddof=1) * math.sqrt(252)
    sharpe = mu / sd if sd > 0 else 0
    eq = (1 + pnl).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    return {"label": label, "n": len(pnl), "ret": mu, "vol": sd, "sharpe": sharpe,
            "max_dd": dd, "hit": (pnl > 0).mean(), "final_eq": eq.iloc[-1]}


def fmt_metrics(m: dict) -> str:
    if m.get("ret") is None:
        return f"  {m['label']:25s} N={m['n']} (insufficient)"
    return (f"  {m['label']:25s} N={m['n']:>5d}  "
            f"ret={m['ret']*100:>5.2f}%  vol={m['vol']*100:>5.2f}%  "
            f"sharpe={m['sharpe']:>5.2f}  DD={m['max_dd']*100:>6.2f}%  "
            f"hit={m['hit']*100:>4.1f}%  eq={m['final_eq']:.2f}x")


def split_metrics(pnl: pd.Series, label: str) -> None:
    pnl = pnl.dropna(); pnl = pnl.loc[pnl.ne(0).idxmax():]
    mid = pnl.index[len(pnl) // 2]
    print(fmt_metrics(metrics(pnl, f"{label} FULL")))
    print(fmt_metrics(metrics(pnl.loc[:mid], f"{label}  IS")))
    print(fmt_metrics(metrics(pnl.loc[mid:], f"{label} OOS")))


def main() -> None:
    fx = load_fx()
    rates = load_rates()

    common = max(fx.index.min(), rates.index.min())
    end = min(fx.index.max(), rates.index.max())
    fx = fx.loc[common:end]
    rates = rates.reindex(fx.index, method="ffill")

    # Carry score: rate differential vs USD (in % points; higher = more attractive long)
    carry_score = rates.drop(columns="USD").subtract(rates["USD"], axis=0)

    # Momentum score: 12-month total return on the FX leg (252 trading days)
    mom_score = fx.pct_change(252)

    # Combined (rank-average) score: average of cross-sectional ranks
    def rank_normalize(df):
        return df.rank(axis=1, method="first") / df.notna().sum(axis=1).values[:, None]
    combo_score = (rank_normalize(carry_score) + rank_normalize(mom_score)) / 2

    print("=" * 72)
    print("CARRY vs MOMENTUM vs COMBINED (vol-targeted, monthly rebalanced, K=2)")
    print("=" * 72)

    pnl_carry = backtest(carry_score, fx, rates, k=2, label="carry")
    pnl_mom   = backtest(mom_score,   fx, rates, k=2, label="mom")
    pnl_combo = backtest(combo_score, fx, rates, k=2, label="combo")

    print()
    split_metrics(pnl_carry, "carry-only")
    print()
    split_metrics(pnl_mom, "12m-mom-only")
    print()
    split_metrics(pnl_combo, "carry+mom")

    # Correlation between the two factors — tells us if they hedge each other
    df = pd.concat([pnl_carry, pnl_mom], axis=1).dropna()
    df.columns = ["carry", "mom"]
    print(f"\n  Correlation carry vs momentum: {df['carry'].corr(df['mom']):.3f}")
    print(f"  Carry+Mom is best when correlation is low/negative (diversification).")

    # Quarterly breakdown of combo
    print("\n=== Combo quarterly breakdown ===")
    q = pnl_combo.dropna().groupby(pd.PeriodIndex(pnl_combo.dropna().index, freq="Y"))
    for p, g in q:
        if len(g) < 10:
            continue
        sh = g.mean() / g.std() * math.sqrt(252) if g.std() > 0 else 0
        print(f"    {str(p):6s} N={len(g):>3d}  ret={g.sum()*100:>6.2f}%  sharpe={sh:>5.2f}")


if __name__ == "__main__":
    main()
