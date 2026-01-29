"""
Backtesting Engine.

Simulates trading with realistic execution using bid/ask prices.
"""

from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional
from uuid import uuid4

import numpy as np
import pandas as pd
from loguru import logger

from fx_trading.config.models import BacktestConfig, CostConfig, RiskConfig
from fx_trading.costs.models import FillCalculator
from fx_trading.data.ingestion import DataIngestor
from fx_trading.execution.paper_broker import PaperBroker
from fx_trading.portfolio.accounting import PortfolioManager
from fx_trading.risk.engine import RiskEngine
from fx_trading.strategies.base import Strategy, StrategyFactory
from fx_trading.types.models import (
    Order,
    OrderType,
    Signal,
    Side,
    PriceData,
    Fill,
)


@dataclass
class BacktestResult:
    """Results from a backtest run."""

    run_id: str
    config: BacktestConfig
    start_time: datetime
    end_time: datetime

    # Performance metrics
    initial_capital: float
    final_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float

    # Trade statistics
    total_trades: int
    winning_trades: int
    losing_trades: int
    win_rate: float
    profit_factor: float
    expectancy: float
    avg_win: float
    avg_loss: float
    max_win: float
    max_loss: float

    # Cost breakdown
    total_commission: float
    total_slippage: float
    total_spread_cost: float

    # DataFrames
    trades_df: pd.DataFrame = field(default_factory=pd.DataFrame)
    equity_curve: pd.DataFrame = field(default_factory=pd.DataFrame)
    signals_log: list[dict] = field(default_factory=list)
    risk_decisions: list[dict] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "run_id": self.run_id,
            "start_time": self.start_time.isoformat(),
            "end_time": self.end_time.isoformat(),
            "initial_capital": self.initial_capital,
            "final_equity": self.final_equity,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "sharpe_ratio": self.sharpe_ratio,
            "sortino_ratio": self.sortino_ratio,
            "total_trades": self.total_trades,
            "winning_trades": self.winning_trades,
            "losing_trades": self.losing_trades,
            "win_rate": self.win_rate,
            "profit_factor": self.profit_factor,
            "expectancy": self.expectancy,
            "avg_win": self.avg_win,
            "avg_loss": self.avg_loss,
            "max_win": self.max_win,
            "max_loss": self.max_loss,
            "total_commission": self.total_commission,
            "total_slippage": self.total_slippage,
            "total_spread_cost": self.total_spread_cost,
        }


class Backtester:
    """
    Backtesting engine for simulating trading strategies.

    Features:
    - Realistic bid/ask execution
    - Spread, slippage, and commission modeling
    - Risk management integration
    - No lookahead bias (signals use only past data)
    - Detailed logging and metrics
    """

    def __init__(
        self,
        config: BacktestConfig,
        strategy: Optional[Strategy] = None,
    ):
        """
        Initialize backtester.

        Args:
            config: Backtest configuration
            strategy: Strategy instance (or created from config)
        """
        self.config = config
        self.run_id = config.run_id or f"bt_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}_{uuid4().hex[:6]}"

        # Initialize components
        self.portfolio = PortfolioManager(
            initial_capital=config.initial_capital,
            base_currency=config.base_currency,
        )

        self.risk_engine = RiskEngine(
            config=config.risk,
            portfolio=self.portfolio,
        )

        self.broker = PaperBroker(
            portfolio=self.portfolio,
            cost_config=config.costs,
            seed=config.random_seed,
        )

        # Create or use provided strategy
        if strategy is not None:
            self.strategy = strategy
        else:
            self.strategy = StrategyFactory.create(config.strategy)

        # Data
        self.data: Optional[pd.DataFrame] = None
        self.ingestor = DataIngestor()

        # Logging
        self.signals_log: list[dict] = []
        self.risk_decisions_log: list[dict] = []
        self.fills_log: list[Fill] = []

        # Set random seed for reproducibility
        if config.random_seed is not None:
            np.random.seed(config.random_seed)

        logger.info(f"Backtester initialized: run_id={self.run_id}")

    def load_data(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Load data from config path, data_dir (multi-symbol), or use provided DataFrame.

        Args:
            data: Optional pre-loaded DataFrame

        Returns:
            Loaded DataFrame
        """
        if data is not None:
            self.data = data
        elif self.config.data_dir is not None:
            # Multi-symbol mode: load from directory
            self.data = self._load_multi_symbol_data()
        elif self.config.data_path is not None:
            self.data = self.ingestor.load_data(
                path=self.config.data_path,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )
        else:
            raise ValueError("Either data_path or data_dir must be specified")

        logger.info(f"Loaded {len(self.data)} bars from {self.data.index[0]} to {self.data.index[-1]}")
        return self.data

    def _load_multi_symbol_data(self) -> pd.DataFrame:
        """
        Load and combine data for multiple symbols from a directory.

        Expects files named {SYMBOL}_M5.parquet in the data_dir.

        Returns:
            Combined DataFrame sorted by timestamp
        """
        from pathlib import Path

        data_dir = Path(self.config.data_dir)
        all_dfs = []

        for symbol in self.config.symbols:
            file_path = data_dir / f"{symbol}_M5.parquet"
            if not file_path.exists():
                logger.warning(f"Data file not found for {symbol}: {file_path}")
                continue

            df = self.ingestor.load_data(
                path=file_path,
                start_date=self.config.start_date,
                end_date=self.config.end_date,
            )

            # Ensure symbol column is set
            df["symbol"] = symbol
            all_dfs.append(df)
            logger.info(f"Loaded {len(df)} bars for {symbol}")

        if not all_dfs:
            raise ValueError(f"No data files found in {data_dir}")

        # Combine all dataframes
        combined = pd.concat(all_dfs, axis=0)

        # Sort by timestamp to interleave all symbols chronologically
        combined = combined.sort_index()

        logger.info(f"Combined {len(combined)} total bars for {len(all_dfs)} symbols")
        return combined

    def run(self, data: Optional[pd.DataFrame] = None) -> BacktestResult:
        """
        Run backtest.

        Args:
            data: Optional pre-loaded data

        Returns:
            BacktestResult with all metrics
        """
        start_time = datetime.utcnow()

        # Load data if not provided
        if data is not None or self.data is None:
            self.load_data(data)

        # Validate data
        if self.data is None or len(self.data) == 0:
            raise ValueError("No data available for backtesting")

        # Connect broker
        self.broker.connect()

        # Get lookback requirement
        lookback = self.strategy.get_lookback_period()

        logger.info(f"Starting backtest: {len(self.data)} bars, lookback={lookback}")

        # Main loop - iterate through each bar
        for i in range(lookback, len(self.data)):
            self._process_bar(i)

        # Close any remaining positions
        self.broker.close_all_positions()

        end_time = datetime.utcnow()

        # Compile results
        result = self._compile_results(start_time, end_time)

        logger.info(
            f"Backtest complete: return={result.total_return_pct:.2f}%, "
            f"trades={result.total_trades}, win_rate={result.win_rate:.1%}"
        )

        return result

    def _process_bar(self, bar_index: int) -> None:
        """
        Process a single bar.

        Args:
            bar_index: Current bar index (iloc position)
        """
        current_bar = self.data.iloc[bar_index]
        current_time = self.data.index[bar_index]

        # Create price data
        price_data = self._create_price_data(current_bar, current_time)

        # Update broker state
        self.broker.set_bar_index(bar_index)
        self.broker.set_prices({price_data.symbol: price_data})

        # Update positions with current prices
        self.broker.update_positions()

        # Check daily reset
        self.portfolio.check_new_day(current_time)

        # Check stop loss and take profit
        fills = self.broker.check_pending_orders()
        self.fills_log.extend(fills)

        # Check time-based exits
        time_fills = self.broker.check_time_exits()
        self.fills_log.extend(time_fills)

        # Record trade results for risk engine
        for fill in fills + time_fills:
            if "net_pnl" in fill.metadata:
                self.risk_engine.record_trade_result(
                    pnl=fill.metadata["net_pnl"],
                    size=fill.size,
                    timestamp=fill.timestamp,
                )

        # Check if kill switch is active
        if self.risk_engine.should_close_all_positions():
            self.broker.close_all_positions()
            return

        # Generate signals (using only past data - no lookahead)
        signals = self.strategy.generate_signals(self.data, bar_index)

        for signal in signals:
            self._process_signal(signal, price_data, current_time, bar_index)

        # Record equity snapshot
        self.portfolio.record_snapshot(current_time)

    def _create_price_data(
        self,
        bar: pd.Series,
        timestamp: datetime,
    ) -> PriceData:
        """Create PriceData from bar data."""
        # Use bid/ask if available, otherwise derive from close
        if "bid" in bar.index and "ask" in bar.index:
            bid = float(bar["bid"])
            ask = float(bar["ask"])
        else:
            # Derive from close using default spread
            close = float(bar["close"])
            half_spread = self.config.costs.default_spread_pips * 0.0001 / 2
            bid = close - half_spread
            ask = close + half_spread

        symbol = bar["symbol"] if "symbol" in bar.index else self.config.symbols[0]

        return PriceData(
            timestamp=timestamp if isinstance(timestamp, datetime) else datetime.utcnow(),
            symbol=symbol,
            bid=bid,
            ask=ask,
            open=float(bar["open"]),
            high=float(bar["high"]),
            low=float(bar["low"]),
            close=float(bar["close"]),
            volume=float(bar.get("volume", 0)),
        )

    def _process_signal(
        self,
        signal: Signal,
        price_data: PriceData,
        current_time: datetime,
        bar_index: int,
    ) -> None:
        """
        Process a trading signal through risk management and execution.

        Args:
            signal: Trading signal
            price_data: Current price data
            current_time: Current timestamp
            bar_index: Current bar index
        """
        # Log signal
        self.signals_log.append({
            "timestamp": current_time,
            "symbol": signal.symbol,
            "side": signal.side.value,
            "strength": signal.strength,
            "stop_loss": signal.stop_loss,
            "take_profit": signal.take_profit,
        })

        # Skip flat signals
        if signal.side == Side.FLAT:
            return

        # Check if we already have a position (strategy should handle this, but double-check)
        existing = self.portfolio.get_open_positions(signal.symbol)
        if existing:
            logger.debug(f"Already have position in {signal.symbol}, skipping signal")
            return

        # Calculate ATR for volatility-based sizing
        volatility = self._calculate_current_volatility(bar_index)

        # Evaluate signal through risk engine
        risk_decision = self.risk_engine.evaluate_signal(
            signal=signal,
            price_data=price_data,
            current_time=current_time,
            volatility=volatility,
        )

        # Log risk decision
        self.risk_decisions_log.append(risk_decision.to_dict())

        if not risk_decision.approved:
            logger.debug(f"Signal rejected: {risk_decision.get_rejection_reasons()}")
            return

        # Create and place order
        order = Order(
            timestamp=current_time,
            symbol=signal.symbol,
            side=signal.side,
            order_type=OrderType.MARKET,
            size=risk_decision.adjusted_size,
            stop_loss=signal.stop_loss,
            take_profit=signal.take_profit,
            time_in_force_bars=signal.time_exit_bars,
        )

        fill = self.broker.place_order(order)

        if fill:
            self.fills_log.append(fill)
            logger.debug(f"Order filled: {signal.symbol} {signal.side.value} @ {fill.fill_price:.5f}")

    def _calculate_current_volatility(self, bar_index: int) -> Optional[float]:
        """Calculate current ATR for volatility."""
        if bar_index < 14:
            return None

        lookback = self.data.iloc[bar_index - 14:bar_index + 1]

        high = lookback["high"]
        low = lookback["low"]
        close = lookback["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        atr = tr.mean()

        return float(atr)

    def _compile_results(
        self,
        start_time: datetime,
        end_time: datetime,
    ) -> BacktestResult:
        """Compile backtest results."""
        metrics = self.portfolio.get_performance_metrics()
        trade_stats = self.portfolio.trade_log.get_stats()

        # Calculate cost totals
        trades = self.portfolio.trade_log.trades
        total_commission = sum(t.total_commission for t in trades)
        total_slippage = sum(t.total_slippage for t in trades)

        # Estimate spread cost
        total_spread_cost = sum(
            t.size * 100000 * self.config.costs.default_spread_pips * 0.0001
            for t in trades
        )

        return BacktestResult(
            run_id=self.run_id,
            config=self.config,
            start_time=start_time,
            end_time=end_time,
            initial_capital=self.config.initial_capital,
            final_equity=self.portfolio.account.equity,
            total_return_pct=metrics.get("total_return_pct", 0),
            max_drawdown_pct=metrics.get("max_drawdown_pct", 0),
            sharpe_ratio=metrics.get("sharpe_ratio", 0),
            sortino_ratio=metrics.get("sortino_ratio", 0),
            total_trades=trade_stats["total_trades"],
            winning_trades=trade_stats["winning_trades"],
            losing_trades=trade_stats["losing_trades"],
            win_rate=trade_stats["win_rate"],
            profit_factor=trade_stats["profit_factor"],
            expectancy=trade_stats["expectancy"],
            avg_win=trade_stats["avg_win"],
            avg_loss=trade_stats["avg_loss"],
            max_win=trade_stats["max_win"],
            max_loss=trade_stats["max_loss"],
            total_commission=total_commission,
            total_slippage=total_slippage,
            total_spread_cost=total_spread_cost,
            trades_df=self.portfolio.trade_log.get_trades_df(),
            equity_curve=self.portfolio.get_equity_curve_df(),
            signals_log=self.signals_log,
            risk_decisions=self.risk_decisions_log,
        )

    def save_results(self, output_dir: Optional[Path] = None) -> Path:
        """
        Save backtest results to files.

        Args:
            output_dir: Output directory (default: config.output_dir/run_id)

        Returns:
            Path to results directory
        """
        import json

        output_dir = output_dir or (self.config.output_dir / self.run_id)
        output_dir.mkdir(parents=True, exist_ok=True)

        # This would be called after run() with the result
        logger.info(f"Results would be saved to {output_dir}")

        return output_dir
