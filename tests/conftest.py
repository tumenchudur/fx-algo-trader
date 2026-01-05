"""Pytest fixtures for FX Trading System tests."""

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fx_trading.config.models import (
    CostConfig,
    RiskConfig,
    StrategyConfig,
    BacktestConfig,
)
from fx_trading.data.synthetic import SyntheticDataGenerator
from fx_trading.portfolio.accounting import PortfolioManager
from fx_trading.types.models import Side, Order, OrderType, PriceData


@pytest.fixture
def sample_data() -> pd.DataFrame:
    """Generate sample OHLCV data for testing."""
    generator = SyntheticDataGenerator(seed=42)
    return generator.generate(
        symbol="EURUSD",
        timeframe="M5",
        num_bars=500,
    )


@pytest.fixture
def small_sample_data() -> pd.DataFrame:
    """Generate small sample data for fast tests."""
    generator = SyntheticDataGenerator(seed=42)
    return generator.generate(
        symbol="EURUSD",
        timeframe="M5",
        num_bars=100,
    )


@pytest.fixture
def cost_config() -> CostConfig:
    """Default cost configuration."""
    return CostConfig(
        commission_per_lot=7.0,
        commission_type="per_lot",
        slippage_model="fixed_pips",
        slippage_pips=0.5,
        use_provided_spread=True,
        default_spread_pips=1.5,
    )


@pytest.fixture
def risk_config() -> RiskConfig:
    """Default risk configuration."""
    return RiskConfig(
        max_risk_per_trade_pct=1.0,
        max_position_size_lots=5.0,
        min_position_size_lots=0.01,
        max_open_positions=3,
        max_exposure_per_currency_pct=30.0,
        daily_loss_limit_pct=3.0,
        max_drawdown_pct=10.0,
        close_positions_on_kill=True,
        max_spread_pips=3.0,
        stale_price_seconds=60.0,
        no_martingale=True,
        no_revenge_trading=True,
    )


@pytest.fixture
def strategy_config() -> StrategyConfig:
    """Default strategy configuration."""
    return StrategyConfig(
        name="test_strategy",
        strategy_type="volatility_breakout",
        enabled=True,
        symbols=["EURUSD"],
        timeframe="M5",
        use_stop_loss=True,
        use_take_profit=True,
        risk_reward_ratio=2.0,
        params={
            "lookback": 20,
            "atr_period": 14,
            "atr_threshold": 1.0,
            "sl_atr_multiplier": 2.0,
        },
    )


@pytest.fixture
def backtest_config(
    cost_config: CostConfig,
    risk_config: RiskConfig,
    strategy_config: StrategyConfig,
    tmp_path: Path,
) -> BacktestConfig:
    """Default backtest configuration."""
    return BacktestConfig(
        run_id="test_backtest",
        output_dir=tmp_path,
        data_path=tmp_path / "data.parquet",
        symbols=["EURUSD"],
        initial_capital=10000.0,
        base_currency="USD",
        strategy=strategy_config,
        costs=cost_config,
        risk=risk_config,
        random_seed=42,
    )


@pytest.fixture
def portfolio() -> PortfolioManager:
    """Fresh portfolio manager."""
    return PortfolioManager(initial_capital=10000.0)


@pytest.fixture
def sample_order() -> Order:
    """Sample order for testing."""
    return Order(
        symbol="EURUSD",
        side=Side.LONG,
        order_type=OrderType.MARKET,
        size=0.1,
        stop_loss=1.0750,
        take_profit=1.0850,
    )


@pytest.fixture
def sample_price_data() -> PriceData:
    """Sample price data for testing."""
    return PriceData(
        timestamp=datetime.utcnow(),
        symbol="EURUSD",
        bid=1.0795,
        ask=1.0797,
        open=1.0790,
        high=1.0800,
        low=1.0785,
        close=1.0796,
    )


@pytest.fixture
def wide_spread_price_data() -> PriceData:
    """Price data with wide spread for testing filters."""
    return PriceData(
        timestamp=datetime.utcnow(),
        symbol="EURUSD",
        bid=1.0790,
        ask=1.0800,  # 10 pip spread
        open=1.0790,
        high=1.0800,
        low=1.0785,
        close=1.0795,
    )


@pytest.fixture
def stale_price_data() -> PriceData:
    """Stale price data for testing filters."""
    return PriceData(
        timestamp=datetime.utcnow() - timedelta(minutes=5),
        symbol="EURUSD",
        bid=1.0795,
        ask=1.0797,
        open=1.0790,
        high=1.0800,
        low=1.0785,
        close=1.0796,
    )
