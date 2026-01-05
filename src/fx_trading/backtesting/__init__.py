"""Backtesting engine and walk-forward evaluation."""

from fx_trading.backtesting.engine import Backtester, BacktestResult
from fx_trading.backtesting.walkforward import WalkForwardAnalysis, WalkForwardResult

__all__ = [
    "Backtester",
    "BacktestResult",
    "WalkForwardAnalysis",
    "WalkForwardResult",
]
