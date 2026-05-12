"""
Vol-targeted FX carry basket.

Construction (daily rebalanced):
  1. For each currency C in {USD, EUR, GBP, JPY, AUD, NZD, CAD, CHF}, take its policy rate.
  2. Express each currency's value vs USD using a synthetic spot:
       - For pair P=CCYUSD (e.g. EURUSD), USD-value-of-CCY = close.
       - For pair P=USDCCY (e.g. USDJPY), USD-value-of-CCY = 1/close.
  3. Rank currencies by yield (higher = "expensive", attractive carry).
  4. Long top K, short bottom K (we use K=3, ignore USD as the funding currency
     since most pairs are USD-crosses).
  5. Each leg is sized to equal *risk* via a 60-day realized vol of daily log returns.
  6. The position generates daily PnL = sum of (size_i * daily_log_return_i)
     plus the *carry* component = sum of (size_i * (rate_i - rate_USD) / 252).
  7. Apply realistic transaction cost: 0.5 pip per leg per rebalance day where
     the position changes. Carry is rebalanced monthly to reduce churn.

Outputs:
  - Aggregated equity curve and metrics (Sharpe, max DD, hit rate, IS/OOS).
  - Per-quarter PnL breakdown to detect regime breaks.
  - Decomposition: how much PnL from carry vs spot moves.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd

DATA_FX = Path("data/yf_d1")
DATA_RATES = Path("data/rates")

# Define each currency's USD-value series construction (pair name + invert flag)
CCY_PAIR = {
    "EUR": ("EURUSD", False),
    "GBP": ("GBPUSD", False),
    "AUD": ("AUDUSD", False),
    "NZD": ("NZDUSD", False),
    "JPY": ("USDJPY", True),
    "CHF": ("USDCHF", True),
    "CAD": ("USDCAD", True),
    # USD is the funding currency: it is implicitly long when we are short the others
    # and short when we are long the others. We don't trade USDUSD directly.
}

# Realistic round-trip cost in pips per leg per rebalance day if the leg changes
COST_PIPS_PER_LEG = {
    "EUR": 0.6, "GBP": 0.8, "JPY": 0.8, "CHF": 1.5, "CAD": 1.0, "AUD": 0.8, "NZD": 1.2
}
PIP_SIZE = {
    "EURUSD": 1e-4, "GBPUSD": 1e-4, "AUDUSD": 1e-4, "NZDUSD": 1e-4,
    "USDJPY": 1e-2, "USDCHF": 1e-4, "USDCAD": 1e-4
}


def load_fx() -> pd.DataFrame:
    """Return DataFrame indexed by date with columns = each currency's USD-value."""
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


def realized_vol(returns: pd.DataFrame, window: int = 60) -> pd.DataFrame:
    return returns.rolling(window).std() * math.sqrt(252)


def carry_signal(rates: pd.DataFrame, k: int) -> pd.DataFrame:
    """Per-day +1 (long top-K), -1 (short bottom-K), 0 otherwise."""
    sig = pd.DataFrame(0, index=rates.index, columns=rates.columns)
    # Use only the non-USD currencies for ranking (USD is implicit)
    rate_no_usd = rates.drop(columns="USD")
    for date, row in rate_no_usd.iterrows():
        ranked = row.dropna().rank(method="first")
        n = len(ranked)
        if n < 2 * k:
            continue
        top = ranked[ranked > n - k].index   # K highest yielders
        bot = ranked[ranked <= k].index       # K lowest yielders
        sig.loc[date, top] = 1
        sig.loc[date, bot] = -1
    return sig.drop(columns="USD")


def run_carry_backtest(k: int = 2, vol_window: int = 60,
                       rebalance_freq: str = "ME", target_vol: float = 0.10):
    """
    Run vol-targeted carry basket. Monthly rebalanced; positions held flat between.

    target_vol: target annualized portfolio vol (e.g. 0.10 = 10%).
    """
    fx = load_fx()
    rates = load_rates()

    common_start = max(fx.index.min(), rates.index.min())
    common_end = min(fx.index.max(), rates.index.max())
    fx = fx.loc[common_start:common_end]
    rates = rates.reindex(fx.index, method="ffill")

    log_ret = np.log(fx / fx.shift(1))
    carry_daily = (rates.drop(columns="USD").subtract(rates["USD"], axis=0)) / 100.0 / 252.0

    raw_sig_full = carry_signal(rates, k=k)
    realized_legs_vol = log_ret.rolling(vol_window).std() * math.sqrt(252)

    # Mark rebalance days = first trading day of each month
    is_rebal = fx.index.to_series().groupby(fx.index.to_period("M")).transform(
        lambda s: s == s.iloc[0]
    ).values

    weights = pd.DataFrame(0.0, index=fx.index, columns=raw_sig_full.columns)
    current_w = pd.Series(0.0, index=raw_sig_full.columns)

    for i, d in enumerate(fx.index):
        if is_rebal[i] and d in raw_sig_full.index:
            s = raw_sig_full.loc[d]
            v = realized_legs_vol.iloc[i]
            if v.isna().any():
                weights.iloc[i] = current_w.values
                continue
            leg_w = (s / v.replace(0, np.nan)).fillna(0)
            gross = leg_w.abs().sum()
            if gross == 0:
                weights.iloc[i] = current_w.values
                continue
            leg_w = leg_w / gross
            approx_basket_vol = float((leg_w.abs() * v).pow(2).sum() ** 0.5)
            if approx_basket_vol > 0:
                leg_w = leg_w * (target_vol / approx_basket_vol)
            current_w = leg_w
        weights.iloc[i] = current_w.values

    # Daily PnL — lagged weights to avoid lookahead
    spot_pnl = (weights.shift(1) * log_ret).sum(axis=1).fillna(0)
    carry_pnl = (weights.shift(1) * carry_daily).sum(axis=1).fillna(0)
    gross_pnl = spot_pnl + carry_pnl

    # Transaction cost: flat 1 bp (0.0001) per absolute weight change per leg.
    # Realistic: 0.5-1.0 pip on majors translates to ~0.5-1.0 bp of notional.
    COST_BP_PER_TURNOVER = 0.0001
    turnover = weights.diff().abs().fillna(0)
    cost_drag = (turnover * COST_BP_PER_TURNOVER).sum(axis=1).fillna(0)
    net_pnl = gross_pnl - cost_drag

    # Equity curve
    equity = (1 + net_pnl.fillna(0)).cumprod()
    return dict(
        net_pnl=net_pnl, gross_pnl=gross_pnl, spot_pnl=spot_pnl, carry_pnl=carry_pnl,
        cost_drag=cost_drag, equity=equity, weights=weights, raw_sig=raw_sig_full,
    )


def metrics(pnl: pd.Series, label: str) -> None:
    pnl = pnl.dropna()
    if len(pnl) < 30:
        print(f"  {label}: insufficient data"); return
    mu = pnl.mean() * 252
    sd = pnl.std(ddof=1) * math.sqrt(252)
    sharpe = mu / sd if sd > 0 else 0
    eq = (1 + pnl).cumprod()
    dd = (eq / eq.cummax() - 1).min()
    hit = (pnl > 0).mean()
    cagr = eq.iloc[-1] ** (252 / len(pnl)) - 1
    print(f"  {label:18s} N={len(pnl):>5d}  ann_return={mu*100:>6.2f}%  ann_vol={sd*100:>6.2f}%  "
          f"sharpe={sharpe:>5.2f}  max_DD={dd*100:>6.2f}%  hit%={hit*100:>5.1f}  CAGR={cagr*100:>5.2f}%")


def quarterly_breakdown(pnl: pd.Series) -> None:
    yq = pnl.groupby(pd.PeriodIndex(pnl.index, freq="Q"))
    rows = []
    for p, g in yq:
        if len(g) < 10: continue
        rows.append((str(p), len(g), g.sum() * 100, g.mean() / g.std() * math.sqrt(252) if g.std() > 0 else 0))
    print("\n  Quarterly: pct_return, sharpe")
    for r in rows:
        print(f"    {r[0]:8s} N={r[1]:>3d}  ret={r[2]:>6.2f}%  sharpe={r[3]:>5.2f}")


def main() -> None:
    print("=" * 72)
    print("FX CARRY BASKET — vol-targeted, monthly rebalanced")
    print("=" * 72)

    for k in [2, 3]:
        print(f"\n>>> Top/Bottom K={k} <<<")
        out = run_carry_backtest(k=k, target_vol=0.10)
        net = out["net_pnl"]; gross = out["gross_pnl"]
        spot = out["spot_pnl"]; carry = out["carry_pnl"]
        first_valid = net.first_valid_index()
        net = net.loc[first_valid:]; gross = gross.loc[first_valid:]
        spot = spot.loc[first_valid:]; carry = carry.loc[first_valid:]

        # Full-sample
        print("\n=== Full sample (target_vol=10%) ===")
        metrics(net, "carry NET")
        metrics(gross, "carry GROSS")
        metrics(spot, "spot leg only")
        metrics(carry, "carry leg only")

        # IS / OOS halves
        mid = net.index[len(net) // 2]
        print("\n=== IS / OOS split ===")
        metrics(net.loc[:mid], "IS net")
        metrics(net.loc[mid:], "OOS net")

        # Equity curve summary
        eq = (1 + net.fillna(0)).cumprod()
        print(f"\n  Final equity multiple: {eq.iloc[-1]:.3f}x  (over {(eq.index[-1] - eq.index[0]).days / 365.25:.1f} years)")

        # Quarterly
        quarterly_breakdown(net)


if __name__ == "__main__":
    main()
