"""Tests for risk engine."""

from datetime import datetime, timedelta

import pytest

from fx_trading.config.models import RiskConfig
from fx_trading.portfolio.accounting import PortfolioManager
from fx_trading.risk.engine import RiskEngine
from fx_trading.types.models import Signal, Side, PriceData


class TestRiskEngineBasic:
    """Basic risk engine tests."""

    def test_approves_valid_signal(self, risk_config, sample_price_data):
        """Should approve signal that passes all checks."""
        portfolio = PortfolioManager(initial_capital=10000.0)
        engine = RiskEngine(risk_config, portfolio)

        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            side=Side.LONG,
            stop_loss=1.0750,
        )

        decision = engine.evaluate_signal(
            signal=signal,
            price_data=sample_price_data,
            current_time=datetime.utcnow(),
        )

        assert decision.approved
        assert decision.adjusted_size > 0

    def test_rejects_flat_signal(self, risk_config, sample_price_data):
        """Should pass through flat signals without rejection."""
        portfolio = PortfolioManager(initial_capital=10000.0)
        engine = RiskEngine(risk_config, portfolio)

        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            side=Side.FLAT,
        )

        decision = engine.evaluate_signal(
            signal=signal,
            price_data=sample_price_data,
            current_time=datetime.utcnow(),
        )

        assert decision.approved


class TestSpreadFilter:
    """Tests for spread filter."""

    def test_rejects_wide_spread(self, risk_config, wide_spread_price_data):
        """Should reject when spread exceeds limit."""
        portfolio = PortfolioManager(initial_capital=10000.0)
        risk_config.max_spread_pips = 3.0  # 10 pip spread will exceed
        engine = RiskEngine(risk_config, portfolio)

        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            side=Side.LONG,
            stop_loss=1.0750,
        )

        decision = engine.evaluate_signal(
            signal=signal,
            price_data=wide_spread_price_data,
            current_time=datetime.utcnow(),
        )

        assert not decision.approved
        assert any("spread" in r.lower() for r in decision.get_rejection_reasons())


class TestStalePriceFilter:
    """Tests for stale price filter."""

    def test_rejects_stale_price(self, risk_config, stale_price_data):
        """Should reject when price is stale."""
        portfolio = PortfolioManager(initial_capital=10000.0)
        risk_config.stale_price_seconds = 60.0  # Price is 5 minutes old
        engine = RiskEngine(risk_config, portfolio)

        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            side=Side.LONG,
            stop_loss=1.0750,
        )

        decision = engine.evaluate_signal(
            signal=signal,
            price_data=stale_price_data,
            current_time=datetime.utcnow(),
        )

        assert not decision.approved
        # Check for stale price rejection (reason contains "old" or check name is "stale_price")
        stale_check = next((c for c in decision.checks if c.check_name == "stale_price"), None)
        assert stale_check is not None
        assert not stale_check.passed


class TestMaxPositions:
    """Tests for max positions limit."""

    def test_blocks_when_max_positions_reached(self, risk_config, sample_price_data):
        """Should block new trades when max positions reached."""
        portfolio = PortfolioManager(initial_capital=10000.0)
        risk_config.max_open_positions = 2
        engine = RiskEngine(risk_config, portfolio)

        # Open 2 positions
        portfolio.open_position(
            symbol="EURUSD",
            side=Side.LONG,
            size=0.1,
            entry_price=1.0800,
            entry_time=datetime.utcnow(),
            entry_bar_index=0,
        )
        portfolio.open_position(
            symbol="GBPUSD",
            side=Side.LONG,
            size=0.1,
            entry_price=1.2600,
            entry_time=datetime.utcnow(),
            entry_bar_index=0,
        )

        # Try to open third
        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="USDJPY",
            side=Side.LONG,
            stop_loss=149.0,
        )

        usdjpy_price = PriceData(
            timestamp=datetime.utcnow(),
            symbol="USDJPY",
            bid=150.00,
            ask=150.02,
        )

        decision = engine.evaluate_signal(
            signal=signal,
            price_data=usdjpy_price,
            current_time=datetime.utcnow(),
        )

        assert not decision.approved
        assert any("position" in r.lower() for r in decision.get_rejection_reasons())


class TestExistingPosition:
    """Tests for existing position check."""

    def test_blocks_duplicate_symbol(self, risk_config, sample_price_data):
        """Should block opening another position in same symbol."""
        portfolio = PortfolioManager(initial_capital=10000.0)
        engine = RiskEngine(risk_config, portfolio)

        # Open a position
        portfolio.open_position(
            symbol="EURUSD",
            side=Side.LONG,
            size=0.1,
            entry_price=1.0800,
            entry_time=datetime.utcnow(),
            entry_bar_index=0,
        )

        # Try to open another in same symbol
        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            side=Side.LONG,
            stop_loss=1.0750,
        )

        decision = engine.evaluate_signal(
            signal=signal,
            price_data=sample_price_data,
            current_time=datetime.utcnow(),
        )

        assert not decision.approved


class TestKillSwitch:
    """Tests for kill switch."""

    def test_kill_switch_blocks_all_trades(self, risk_config, sample_price_data):
        """Kill switch should block all trades."""
        portfolio = PortfolioManager(initial_capital=10000.0)
        engine = RiskEngine(risk_config, portfolio)

        # Manually activate kill switch
        engine._activate_kill_switch("Test kill", datetime.utcnow())

        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            side=Side.LONG,
            stop_loss=1.0750,
        )

        decision = engine.evaluate_signal(
            signal=signal,
            price_data=sample_price_data,
            current_time=datetime.utcnow(),
        )

        assert not decision.approved
        assert decision.kill_switch_active

    def test_drawdown_triggers_kill_switch(self, risk_config, sample_price_data):
        """Max drawdown should trigger kill switch."""
        portfolio = PortfolioManager(initial_capital=10000.0)
        risk_config.max_drawdown_pct = 5.0
        engine = RiskEngine(risk_config, portfolio)

        # Simulate 6% drawdown
        portfolio.account.balance = 9400
        portfolio.account.peak_equity = 10000
        portfolio.account.update(0, 0)

        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            side=Side.LONG,
            stop_loss=1.0750,
        )

        decision = engine.evaluate_signal(
            signal=signal,
            price_data=sample_price_data,
            current_time=datetime.utcnow(),
        )

        assert not decision.approved
        assert engine.kill_switch_active


class TestDailyLossLimit:
    """Tests for daily loss limit."""

    def test_daily_loss_triggers_kill_switch(self, risk_config, sample_price_data):
        """Daily loss limit should trigger kill switch."""
        portfolio = PortfolioManager(initial_capital=10000.0)
        risk_config.daily_loss_limit_pct = 2.0
        engine = RiskEngine(risk_config, portfolio)

        # Initialize daily tracking
        current_time = datetime.utcnow()
        engine.current_date = current_time.date()
        engine.daily_start_equity = 10000.0

        # Simulate 2.5% daily loss
        portfolio.account.balance = 9750
        portfolio.account.update(0, 0)

        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            side=Side.LONG,
            stop_loss=1.0750,
        )

        decision = engine.evaluate_signal(
            signal=signal,
            price_data=sample_price_data,
            current_time=current_time,
        )

        assert not decision.approved
        assert engine.kill_switch_active


class TestPositionSizing:
    """Tests for position sizing in risk engine."""

    def test_calculates_position_size(self, risk_config, sample_price_data):
        """Should calculate appropriate position size."""
        portfolio = PortfolioManager(initial_capital=10000.0)
        engine = RiskEngine(risk_config, portfolio)

        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            side=Side.LONG,
            stop_loss=1.0750,  # ~45 pips from ask at 1.0797
        )

        decision = engine.evaluate_signal(
            signal=signal,
            price_data=sample_price_data,
            current_time=datetime.utcnow(),
        )

        assert decision.approved
        assert decision.adjusted_size is not None
        assert decision.adjusted_size >= risk_config.min_position_size_lots
        assert decision.adjusted_size <= risk_config.max_position_size_lots

    def test_respects_max_size_limit(self, risk_config, sample_price_data):
        """Position size should not exceed max limit."""
        portfolio = PortfolioManager(initial_capital=1000000.0)  # Large capital
        risk_config.max_position_size_lots = 5.0
        engine = RiskEngine(risk_config, portfolio)

        signal = Signal(
            timestamp=datetime.utcnow(),
            symbol="EURUSD",
            side=Side.LONG,
            stop_loss=1.0795,  # Very tight stop = large calculated size
        )

        decision = engine.evaluate_signal(
            signal=signal,
            price_data=sample_price_data,
            current_time=datetime.utcnow(),
        )

        assert decision.adjusted_size <= risk_config.max_position_size_lots
