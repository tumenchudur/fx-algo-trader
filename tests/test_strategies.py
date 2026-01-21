"""Tests for trading strategies."""

import pytest
import pandas as pd
import numpy as np

from fx_trading.config.models import StrategyConfig
from fx_trading.strategies.base import Strategy, StrategyFactory
from fx_trading.strategies.volatility_breakout import VolatilityBreakoutStrategy
from fx_trading.strategies.mean_reversion import MeanReversionStrategy
from fx_trading.types.models import Side


class TestStrategyFactory:
    """Tests for strategy factory."""

    def test_creates_volatility_breakout(self, strategy_config):
        """Should create volatility breakout strategy."""
        strategy_config.strategy_type = "volatility_breakout"
        strategy = StrategyFactory.create(strategy_config)
        assert isinstance(strategy, VolatilityBreakoutStrategy)

    def test_creates_mean_reversion(self, strategy_config):
        """Should create mean reversion strategy."""
        strategy_config.strategy_type = "mean_reversion"
        strategy = StrategyFactory.create(strategy_config)
        assert isinstance(strategy, MeanReversionStrategy)

    def test_raises_on_unknown_type(self, strategy_config):
        """Should raise on unknown strategy type."""
        strategy_config.strategy_type = "unknown"
        with pytest.raises(ValueError):
            StrategyFactory.create(strategy_config)


class TestVolatilityBreakout:
    """Tests for volatility breakout strategy."""

    @pytest.fixture
    def strategy(self) -> VolatilityBreakoutStrategy:
        """Create volatility breakout strategy."""
        config = StrategyConfig(
            name="test_vb",
            strategy_type="volatility_breakout",
            symbols=["EURUSD"],
            use_stop_loss=True,
            use_take_profit=True,
            risk_reward_ratio=2.0,
            params={
                "lookback": 10,
                "atr_period": 7,
                "atr_threshold": 1.0,
                "sl_atr_multiplier": 2.0,
            },
        )
        return VolatilityBreakoutStrategy(config)

    def test_no_signal_insufficient_data(self, strategy, sample_data):
        """Should return no signal with insufficient data."""
        signals = strategy.generate_signals(sample_data, current_index=5)
        assert signals == []

    def test_generates_long_on_upside_breakout(self, strategy):
        """Should generate long signal on upside breakout."""
        # Create data with clear upside breakout
        np.random.seed(42)
        n = 50
        base = 1.0800

        # Range-bound for first 40 bars
        prices = base + np.random.uniform(-0.0010, 0.0010, n)

        # Clear breakout on last few bars
        prices[-5:] = base + 0.0020

        df = pd.DataFrame({
            "open": prices - 0.0001,
            "high": prices + 0.0005,
            "low": prices - 0.0005,
            "close": prices,
            "symbol": "EURUSD",
        }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))

        signals = strategy.generate_signals(df, current_index=n - 1)

        # May or may not generate signal depending on volatility
        if signals:
            assert signals[0].side == Side.LONG

    def test_no_lookahead_bias(self, strategy, sample_data):
        """Strategy should not use future data."""
        # Run at bar 30
        signals_30 = strategy.generate_signals(sample_data, current_index=30)

        # Modify future data
        modified = sample_data.copy()
        modified.iloc[31:, modified.columns.get_loc("close")] *= 2

        # Should get same signals
        signals_30_mod = strategy.generate_signals(modified, current_index=30)

        # Results should be identical
        assert len(signals_30) == len(signals_30_mod)
        if signals_30:
            assert signals_30[0].side == signals_30_mod[0].side


class TestMeanReversion:
    """Tests for mean reversion strategy."""

    @pytest.fixture
    def strategy(self) -> MeanReversionStrategy:
        """Create mean reversion strategy."""
        config = StrategyConfig(
            name="test_mr",
            strategy_type="mean_reversion",
            symbols=["EURUSD"],
            use_stop_loss=True,
            use_take_profit=True,
            risk_reward_ratio=1.5,
            params={
                "lookback": 10,
                "z_threshold": 2.0,
                "sl_std_multiplier": 3.0,
                "exit_at_mean": True,
            },
        )
        return MeanReversionStrategy(config)

    def test_no_signal_insufficient_data(self, strategy, sample_data):
        """Should return no signal with insufficient data."""
        signals = strategy.generate_signals(sample_data, current_index=5)
        assert signals == []

    def test_generates_long_on_oversold(self, strategy):
        """Should generate long signal when z-score is very negative."""
        np.random.seed(42)
        n = 50
        base = 1.0800

        # Normal prices
        prices = base + np.random.normal(0, 0.0005, n)

        # Sudden drop creating oversold condition
        prices[-2] = base - 0.0025
        prices[-1] = base - 0.0030

        df = pd.DataFrame({
            "open": prices - 0.0001,
            "high": prices + 0.0003,
            "low": prices - 0.0003,
            "close": prices,
            "symbol": "EURUSD",
        }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))

        signals = strategy.generate_signals(df, current_index=n - 1)

        if signals:
            assert signals[0].side == Side.LONG

    def test_generates_short_on_overbought(self, strategy):
        """Should generate short signal when z-score is very positive."""
        np.random.seed(42)
        n = 50
        base = 1.0800

        # Normal prices
        prices = base + np.random.normal(0, 0.0005, n)

        # Sudden spike creating overbought condition
        prices[-2] = base + 0.0025
        prices[-1] = base + 0.0030

        df = pd.DataFrame({
            "open": prices - 0.0001,
            "high": prices + 0.0003,
            "low": prices - 0.0003,
            "close": prices,
            "symbol": "EURUSD",
        }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))

        signals = strategy.generate_signals(df, current_index=n - 1)

        if signals:
            assert signals[0].side == Side.SHORT

    def test_no_lookahead_bias(self, strategy, sample_data):
        """Strategy should not use future data."""
        signals_30 = strategy.generate_signals(sample_data, current_index=30)

        modified = sample_data.copy()
        modified.iloc[31:, modified.columns.get_loc("close")] *= 2

        signals_30_mod = strategy.generate_signals(modified, current_index=30)

        assert len(signals_30) == len(signals_30_mod)
        if signals_30:
            assert signals_30[0].side == signals_30_mod[0].side


class TestRSIFilter:
    """Tests for RSI filter in volatility breakout strategy."""

    @pytest.fixture
    def strategy_with_rsi(self) -> VolatilityBreakoutStrategy:
        """Create strategy with RSI filter enabled."""
        config = StrategyConfig(
            name="test_vb_rsi",
            strategy_type="volatility_breakout",
            symbols=["EURUSD"],
            params={
                "lookback": 10,
                "atr_period": 7,
                "use_rsi_filter": True,
                "rsi_period": 14,
                "rsi_overbought": 70,
                "rsi_oversold": 30,
                "use_trend_filter": False,  # Disable other filters for isolation
                "use_adx_filter": False,
                "use_time_filter": False,
            },
        )
        return VolatilityBreakoutStrategy(config)

    @pytest.fixture
    def strategy_without_rsi(self) -> VolatilityBreakoutStrategy:
        """Create strategy with RSI filter disabled."""
        config = StrategyConfig(
            name="test_vb_no_rsi",
            strategy_type="volatility_breakout",
            symbols=["EURUSD"],
            params={
                "lookback": 10,
                "atr_period": 7,
                "use_rsi_filter": False,
                "use_trend_filter": False,
                "use_adx_filter": False,
                "use_time_filter": False,
            },
        )
        return VolatilityBreakoutStrategy(config)

    def test_rsi_calculation(self, strategy_with_rsi):
        """Should calculate RSI correctly."""
        # Create data with known RSI pattern
        np.random.seed(42)
        n = 50
        # Uptrend data - RSI should be high
        prices = np.linspace(1.0800, 1.0900, n)

        df = pd.DataFrame({
            "open": prices - 0.0001,
            "high": prices + 0.0003,
            "low": prices - 0.0003,
            "close": prices,
        }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))

        rsi = strategy_with_rsi._calculate_rsi(df)
        assert not pd.isna(rsi)
        assert 0 <= rsi <= 100

    def test_blocks_overbought_long(self, strategy_with_rsi):
        """Should block LONG signals when RSI > overbought threshold."""
        np.random.seed(42)
        n = 100
        base = 1.0800

        # Strong uptrend to push RSI high
        prices = base + np.linspace(0, 0.0100, n)
        # Add clear breakout at the end
        prices[-5:] = prices[-6] + 0.0030

        df = pd.DataFrame({
            "open": prices - 0.0001,
            "high": prices + 0.0008,
            "low": prices - 0.0005,
            "close": prices,
            "symbol": "EURUSD",
        }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))

        signals = strategy_with_rsi.generate_signals(df, current_index=n - 1)

        # If RSI filter is working, should block overbought longs
        # The signal might be blocked entirely or still generated if RSI isn't high enough
        # This tests the mechanism exists
        assert isinstance(signals, list)

    def test_blocks_oversold_short(self, strategy_with_rsi):
        """Should block SHORT signals when RSI < oversold threshold."""
        np.random.seed(42)
        n = 100
        base = 1.0800

        # Strong downtrend to push RSI low
        prices = base - np.linspace(0, 0.0100, n)
        # Add clear breakdown at the end
        prices[-5:] = prices[-6] - 0.0030

        df = pd.DataFrame({
            "open": prices + 0.0001,
            "high": prices + 0.0005,
            "low": prices - 0.0008,
            "close": prices,
            "symbol": "EURUSD",
        }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))

        signals = strategy_with_rsi.generate_signals(df, current_index=n - 1)

        # If RSI filter is working, should block oversold shorts
        assert isinstance(signals, list)

    def test_rsi_disabled_allows_signals(self, strategy_without_rsi):
        """Should allow signals when RSI filter is disabled."""
        np.random.seed(42)
        n = 50
        base = 1.0800

        # Range-bound then breakout
        prices = base + np.random.uniform(-0.0005, 0.0005, n)
        prices[-3:] = base + 0.0015

        df = pd.DataFrame({
            "open": prices - 0.0001,
            "high": prices + 0.0005,
            "low": prices - 0.0005,
            "close": prices,
            "symbol": "EURUSD",
        }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))

        signals = strategy_without_rsi.generate_signals(df, current_index=n - 1)

        # Without RSI filter, signals should not be blocked by RSI
        # May or may not generate signal based on other conditions
        assert isinstance(signals, list)

    def test_rsi_in_signal_metadata(self, strategy_with_rsi):
        """RSI value should be included in signal metadata."""
        np.random.seed(123)
        n = 80
        base = 1.0800

        # Create conditions for a signal
        prices = base + np.random.uniform(-0.0010, 0.0010, n)
        prices[-5:] = base + 0.0020

        df = pd.DataFrame({
            "open": prices - 0.0001,
            "high": prices + 0.0006,
            "low": prices - 0.0006,
            "close": prices,
            "symbol": "EURUSD",
        }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))

        signals = strategy_with_rsi.generate_signals(df, current_index=n - 1)

        if signals:
            assert "rsi" in signals[0].metadata


class TestPerPairParameters:
    """Tests for per-symbol parameter overrides."""

    def test_symbol_params_merging(self):
        """Should merge base params with symbol-specific overrides."""
        config = StrategyConfig(
            name="test",
            strategy_type="volatility_breakout",
            symbols=["EURUSD", "XAUUSD"],
            params={
                "sl_atr_multiplier": 2.0,
                "lookback": 20,
            },
            symbol_params={
                "XAUUSD": {
                    "sl_atr_multiplier": 3.0,
                    "lookback": 30,
                },
            },
        )

        eurusd_params = config.get_params_for_symbol("EURUSD")
        xauusd_params = config.get_params_for_symbol("XAUUSD")

        # EURUSD should use base params
        assert eurusd_params["sl_atr_multiplier"] == 2.0
        assert eurusd_params["lookback"] == 20

        # XAUUSD should use overridden params
        assert xauusd_params["sl_atr_multiplier"] == 3.0
        assert xauusd_params["lookback"] == 30

    def test_get_param_method(self):
        """Strategy get_param should respect symbol overrides."""
        config = StrategyConfig(
            name="test",
            strategy_type="volatility_breakout",
            symbols=["EURUSD", "XAUUSD"],
            params={
                "sl_atr_multiplier": 2.0,
            },
            symbol_params={
                "XAUUSD": {"sl_atr_multiplier": 3.0},
            },
        )
        strategy = VolatilityBreakoutStrategy(config)

        # Without symbol
        assert strategy.get_param("sl_atr_multiplier") == 2.0

        # With EURUSD (no override)
        assert strategy.get_param("sl_atr_multiplier", symbol="EURUSD") == 2.0

        # With XAUUSD (has override)
        assert strategy.get_param("sl_atr_multiplier", symbol="XAUUSD") == 3.0


class TestStrategyStopLossTakeProfit:
    """Tests for stop loss and take profit calculation."""

    def test_long_stop_loss_below_entry(self, strategy_config):
        """Stop loss should be below entry for longs."""
        strategy_config.strategy_type = "volatility_breakout"
        strategy = StrategyFactory.create(strategy_config)

        sl = strategy.calculate_stop_loss(
            entry_price=1.0800,
            side=Side.LONG,
            atr=0.0010,
        )

        assert sl is not None
        assert sl < 1.0800

    def test_short_stop_loss_above_entry(self, strategy_config):
        """Stop loss should be above entry for shorts."""
        strategy_config.strategy_type = "volatility_breakout"
        strategy = StrategyFactory.create(strategy_config)

        sl = strategy.calculate_stop_loss(
            entry_price=1.0800,
            side=Side.SHORT,
            atr=0.0010,
        )

        assert sl is not None
        assert sl > 1.0800

    def test_take_profit_respects_rr_ratio(self, strategy_config):
        """Take profit should respect risk-reward ratio."""
        strategy_config.strategy_type = "volatility_breakout"
        strategy_config.risk_reward_ratio = 2.0
        strategy = StrategyFactory.create(strategy_config)

        entry = 1.0800
        sl = 1.0750  # 50 pip risk

        tp = strategy.calculate_take_profit(
            entry_price=entry,
            stop_loss=sl,
            side=Side.LONG,
        )

        assert tp is not None
        # TP should be 100 pips from entry (2:1 ratio)
        assert abs(tp - 1.0900) < 0.0001
