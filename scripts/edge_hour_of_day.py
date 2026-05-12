"""
Hour-of-day return analysis on real FX data.

For each symbol, compute the mean log-return per UTC hour, with a t-stat.
This is the cleanest test of intraday session bias. We DETREND first
(subtract overall mean) so a directional drift in the sample doesn't
masquerade as an "edge".

If a real intraday pattern exists, some hours should show t > 2 in BOTH
the in-sample and out-of-sample halves.
"""

from __future__ import annotations

import glob
import math
from pathlib import Path

import numpy as np
import pandas as pd

PIP_SIZE = {"EURUSD": 1e-4, "GBPUSD": 1e-4, "USDJPY": 1e-2, "XAUUSD": 1e-2}


def load_all() -> dict[str, pd.DataFrame]:
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
    out = {}
    for sym, frames in bag.items():
        merged = pd.concat(frames).sort_index()
        merged = merged[~merged.index.duplicated(keep="first")]
        out[sym] = merged[["open", "close"]].astype(float)
    return out


def hourly_table(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    """For each UTC hour, compute mean *detrended* log-return per bar with t-stat."""
    pip = PIP_SIZE[symbol]
    r = np.log(df["close"] / df["close"].shift(1)).dropna()
    r_pips = (df["close"].diff() / pip).dropna()  # absolute pip move

    # Detrend the pip return so we measure deviation from sample drift.
    r_pips_detrended = r_pips - r_pips.mean()

    tbl = pd.DataFrame({
        "hour": r_pips_detrended.index.hour,
        "ret_pips": r_pips_detrended.values,
    })
    g = tbl.groupby("hour")["ret_pips"]
    out = pd.DataFrame({
        "n": g.count(),
        "mean_pips": g.mean(),
        "std": g.std(ddof=1),
    })
    out["t_stat"] = out["mean_pips"] / (out["std"] / np.sqrt(out["n"]))
    return out


def split(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    mid = df.index[len(df) // 2]
    return df[df.index < mid], df[df.index >= mid]


def run() -> None:
    data = load_all()
    for symbol, df in data.items():
        is_df, oos_df = split(df)
        is_tbl = hourly_table(is_df, symbol)
        oos_tbl = hourly_table(oos_df, symbol)
        merged = is_tbl.join(oos_tbl, lsuffix="_is", rsuffix="_oos")

        print(f"\n=== {symbol} | rows={len(df)} | span {df.index.min().date()} -> {df.index.max().date()} ===")
        # Show hour, mean detrended pips IS / OOS, t-stats. Bold pattern: same sign + |t|>2 BOTH halves.
        merged["consistent"] = (
            (np.sign(merged["mean_pips_is"]) == np.sign(merged["mean_pips_oos"]))
            & (merged["t_stat_is"].abs() > 2.0)
            & (merged["t_stat_oos"].abs() > 2.0)
        )
        cols = ["mean_pips_is", "t_stat_is", "mean_pips_oos", "t_stat_oos", "consistent"]
        print(merged[cols].round(3).to_string())

        winners = merged[merged["consistent"]]
        if len(winners):
            print(f"  -> {len(winners)} hour(s) with same-sign |t|>2 in BOTH halves: {list(winners.index)}")
        else:
            print("  -> No hour passes the IS+OOS consistency test.")


if __name__ == "__main__":
    run()
