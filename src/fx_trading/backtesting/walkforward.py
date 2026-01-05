"""
Walk-Forward Analysis.

Implements out-of-sample testing with rolling or anchored windows.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from fx_trading.backtesting.engine import Backtester, BacktestResult
from fx_trading.config.models import WalkForwardConfig, BacktestConfig
from fx_trading.data.ingestion import DataIngestor
from fx_trading.strategies.base import StrategyFactory


@dataclass
class WindowResult:
    """Result from a single walk-forward window."""

    window_id: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    train_result: Optional[BacktestResult]
    test_result: BacktestResult

    @property
    def is_train_window(self) -> bool:
        """Check if this includes training."""
        return self.train_result is not None


@dataclass
class WalkForwardResult:
    """Aggregated walk-forward analysis results."""

    run_id: str
    config: WalkForwardConfig
    windows: list[WindowResult] = field(default_factory=list)

    # Aggregated out-of-sample metrics
    total_oos_return_pct: float = 0.0
    avg_oos_return_pct: float = 0.0
    oos_sharpe_ratio: float = 0.0
    oos_max_drawdown_pct: float = 0.0
    total_oos_trades: int = 0
    oos_win_rate: float = 0.0

    # Consistency metrics
    profitable_windows: int = 0
    window_consistency: float = 0.0  # % of profitable windows

    # Combined equity curve
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)

    def add_window(self, result: WindowResult) -> None:
        """Add window result and update aggregates."""
        self.windows.append(result)
        self._update_aggregates()

    def _update_aggregates(self) -> None:
        """Update aggregated metrics."""
        if not self.windows:
            return

        oos_returns = [w.test_result.total_return_pct for w in self.windows]
        oos_sharpes = [w.test_result.sharpe_ratio for w in self.windows]
        oos_drawdowns = [w.test_result.max_drawdown_pct for w in self.windows]
        oos_trades = [w.test_result.total_trades for w in self.windows]
        oos_win_rates = [w.test_result.win_rate for w in self.windows if w.test_result.total_trades > 0]

        self.total_oos_return_pct = sum(oos_returns)
        self.avg_oos_return_pct = sum(oos_returns) / len(oos_returns)
        self.oos_sharpe_ratio = sum(oos_sharpes) / len(oos_sharpes) if oos_sharpes else 0
        self.oos_max_drawdown_pct = max(oos_drawdowns) if oos_drawdowns else 0
        self.total_oos_trades = sum(oos_trades)
        self.oos_win_rate = sum(oos_win_rates) / len(oos_win_rates) if oos_win_rates else 0

        self.profitable_windows = sum(1 for r in oos_returns if r > 0)
        self.window_consistency = self.profitable_windows / len(self.windows)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "num_windows": len(self.windows),
            "total_oos_return_pct": self.total_oos_return_pct,
            "avg_oos_return_pct": self.avg_oos_return_pct,
            "oos_sharpe_ratio": self.oos_sharpe_ratio,
            "oos_max_drawdown_pct": self.oos_max_drawdown_pct,
            "total_oos_trades": self.total_oos_trades,
            "oos_win_rate": self.oos_win_rate,
            "profitable_windows": self.profitable_windows,
            "window_consistency": self.window_consistency,
            "windows": [
                {
                    "window_id": w.window_id,
                    "train_start": w.train_start.isoformat(),
                    "train_end": w.train_end.isoformat(),
                    "test_start": w.test_start.isoformat(),
                    "test_end": w.test_end.isoformat(),
                    "test_return_pct": w.test_result.total_return_pct,
                    "test_trades": w.test_result.total_trades,
                    "test_sharpe": w.test_result.sharpe_ratio,
                }
                for w in self.windows
            ],
        }


class WalkForwardAnalysis:
    """
    Walk-forward analysis engine.

    Implements rolling or anchored walk-forward testing:
    - Split data into train/test windows
    - Run strategy on each test window
    - Aggregate out-of-sample results
    """

    def __init__(self, config: WalkForwardConfig):
        """
        Initialize walk-forward analysis.

        Args:
            config: Walk-forward configuration
        """
        self.config = config
        self.run_id = f"wf_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}"
        self.ingestor = DataIngestor()

        logger.info(
            f"WalkForward initialized: {config.num_windows} windows, "
            f"train_pct={config.train_pct}, anchored={config.anchored}"
        )

    def run(self, data: Optional[pd.DataFrame] = None) -> WalkForwardResult:
        """
        Run walk-forward analysis.

        Args:
            data: Optional pre-loaded data

        Returns:
            WalkForwardResult with all window results
        """
        # Load data
        if data is None:
            data = self.ingestor.load_data(
                path=self.config.base_config.data_path,
                start_date=self.config.base_config.start_date,
                end_date=self.config.base_config.end_date,
            )

        logger.info(f"Running walk-forward on {len(data)} bars")

        # Generate windows
        windows = self._generate_windows(data)

        # Initialize result
        result = WalkForwardResult(
            run_id=self.run_id,
            config=self.config,
        )

        # Process each window
        for i, (train_start, train_end, test_start, test_end) in enumerate(windows):
            logger.info(f"Processing window {i + 1}/{len(windows)}")

            window_result = self._process_window(
                window_id=i,
                data=data,
                train_start=train_start,
                train_end=train_end,
                test_start=test_start,
                test_end=test_end,
            )

            result.add_window(window_result)

        # Combine equity curves
        result.equity_curve = self._combine_equity_curves(result.windows)

        logger.info(
            f"Walk-forward complete: OOS return={result.total_oos_return_pct:.2f}%, "
            f"consistency={result.window_consistency:.1%}"
        )

        return result

    def _generate_windows(
        self,
        data: pd.DataFrame,
    ) -> list[tuple[datetime, datetime, datetime, datetime]]:
        """
        Generate train/test window boundaries.

        Returns:
            List of (train_start, train_end, test_start, test_end) tuples
        """
        if self.config.windows is not None:
            # Use manually specified windows
            return [
                (w.train_start, w.train_end, w.test_start, w.test_end)
                for w in self.config.windows
            ]

        # Generate windows automatically
        total_bars = len(data)
        num_windows = self.config.num_windows
        train_pct = self.config.train_pct
        gap_bars = self.config.gap_bars

        windows = []

        if self.config.anchored:
            # Anchored: training set grows, test set stays same size
            test_size = int(total_bars * (1 - train_pct) / num_windows)

            for i in range(num_windows):
                train_start_idx = 0
                train_end_idx = int(total_bars * train_pct) + (i * test_size)
                test_start_idx = train_end_idx + gap_bars
                test_end_idx = test_start_idx + test_size

                if test_end_idx > total_bars:
                    break

                windows.append((
                    data.index[train_start_idx],
                    data.index[train_end_idx - 1],
                    data.index[test_start_idx],
                    data.index[min(test_end_idx - 1, total_bars - 1)],
                ))
        else:
            # Rolling: fixed train and test size
            window_size = total_bars // num_windows
            train_size = int(window_size * train_pct)
            test_size = window_size - train_size - gap_bars

            for i in range(num_windows):
                start_idx = i * window_size
                train_start_idx = start_idx
                train_end_idx = start_idx + train_size
                test_start_idx = train_end_idx + gap_bars
                test_end_idx = test_start_idx + test_size

                if test_end_idx > total_bars:
                    test_end_idx = total_bars

                if test_start_idx >= total_bars:
                    break

                windows.append((
                    data.index[train_start_idx],
                    data.index[train_end_idx - 1],
                    data.index[test_start_idx],
                    data.index[min(test_end_idx - 1, total_bars - 1)],
                ))

        logger.info(f"Generated {len(windows)} walk-forward windows")
        return windows

    def _process_window(
        self,
        window_id: int,
        data: pd.DataFrame,
        train_start: datetime,
        train_end: datetime,
        test_start: datetime,
        test_end: datetime,
    ) -> WindowResult:
        """
        Process a single walk-forward window.

        Args:
            window_id: Window identifier
            data: Full dataset
            train_start: Training start time
            train_end: Training end time
            test_start: Test start time
            test_end: Test end time

        Returns:
            WindowResult
        """
        # Get test data
        test_data = data[(data.index >= test_start) & (data.index <= test_end)]

        logger.info(
            f"Window {window_id}: train=[{train_start} to {train_end}], "
            f"test=[{test_start} to {test_end}] ({len(test_data)} bars)"
        )

        # Create strategy (would use trained params in real optimization)
        strategy = StrategyFactory.create(self.config.base_config.strategy)

        # Create backtest config for this window
        window_config = BacktestConfig(
            run_id=f"{self.run_id}_w{window_id}",
            output_dir=self.config.base_config.output_dir,
            data_path=self.config.base_config.data_path,
            symbols=self.config.base_config.symbols,
            start_date=test_start,
            end_date=test_end,
            initial_capital=self.config.base_config.initial_capital,
            base_currency=self.config.base_config.base_currency,
            strategy=self.config.base_config.strategy,
            costs=self.config.base_config.costs,
            risk=self.config.base_config.risk,
            random_seed=self.config.base_config.random_seed,
        )

        # Run backtest on test data
        backtester = Backtester(config=window_config, strategy=strategy)
        test_result = backtester.run(data=test_data)

        return WindowResult(
            window_id=window_id,
            train_start=train_start,
            train_end=train_end,
            test_start=test_start,
            test_end=test_end,
            train_result=None,  # Training optimization not implemented
            test_result=test_result,
        )

    def _combine_equity_curves(
        self,
        windows: list[WindowResult],
    ) -> pd.DataFrame:
        """Combine equity curves from all windows."""
        curves = []

        for w in windows:
            if not w.test_result.equity_curve.empty:
                curve = w.test_result.equity_curve.copy()
                curve["window_id"] = w.window_id
                curves.append(curve)

        if curves:
            return pd.concat(curves)
        return pd.DataFrame()

    def save_results(
        self,
        result: WalkForwardResult,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save walk-forward results.

        Args:
            result: Walk-forward result
            output_dir: Output directory

        Returns:
            Path to results directory
        """
        import json

        output_dir = output_dir or (self.config.base_config.output_dir / result.run_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save summary
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Save equity curve
        if not result.equity_curve.empty:
            equity_path = output_dir / "equity_curve.parquet"
            result.equity_curve.to_parquet(equity_path)

        logger.info(f"Results saved to {output_dir}")
        return output_dir
