"""Tests for position sizing."""

import pytest

from fx_trading.config.models import RiskConfig
from fx_trading.portfolio.position_sizing import PositionSizer
from fx_trading.types.models import Side


class TestPositionSizer:
    """Tests for position sizer."""

    @pytest.fixture
    def sizer(self, risk_config) -> PositionSizer:
        """Create position sizer."""
        return PositionSizer(risk_config)

    def test_calculates_size_from_stop_loss(self, sizer):
        """Should calculate size based on stop loss distance."""
        size = sizer.calculate_size(
            equity=10000.0,
            entry_price=1.0800,
            stop_loss=1.0750,  # 50 pips risk
            side=Side.LONG,
            symbol="EURUSD",
        )

        # Risk = 1% of 10000 = 100
        # SL distance = 50 pips = $500 per lot
        # Size = 100 / 500 = 0.2 lots
        assert 0.1 <= size <= 0.3

    def test_respects_min_size(self, sizer, risk_config):
        """Should not go below minimum size."""
        risk_config.min_position_size_lots = 0.01
        sizer = PositionSizer(risk_config)

        size = sizer.calculate_size(
            equity=100.0,  # Very small account
            entry_price=1.0800,
            stop_loss=1.0750,
            side=Side.LONG,
            symbol="EURUSD",
        )

        assert size >= risk_config.min_position_size_lots

    def test_respects_max_size(self, sizer, risk_config):
        """Should not exceed maximum size."""
        risk_config.max_position_size_lots = 5.0
        sizer = PositionSizer(risk_config)

        size = sizer.calculate_size(
            equity=1000000.0,  # Very large account
            entry_price=1.0800,
            stop_loss=1.0799,  # Very tight stop = large size
            side=Side.LONG,
            symbol="EURUSD",
        )

        assert size <= risk_config.max_position_size_lots

    def test_handles_no_stop_loss(self, sizer, risk_config):
        """Should use max size when no stop loss."""
        size = sizer.calculate_size(
            equity=10000.0,
            entry_price=1.0800,
            stop_loss=None,
            side=Side.LONG,
            symbol="EURUSD",
        )

        assert size == risk_config.max_position_size_lots

    def test_handles_short_position(self, sizer):
        """Should calculate correctly for short positions."""
        size = sizer.calculate_size(
            equity=10000.0,
            entry_price=1.0800,
            stop_loss=1.0850,  # 50 pips above for short
            side=Side.SHORT,
            symbol="EURUSD",
        )

        assert size > 0

    def test_rounds_to_micro_lots(self, sizer):
        """Size should be rounded to 0.01 increments."""
        size = sizer.calculate_size(
            equity=10000.0,
            entry_price=1.0800,
            stop_loss=1.0750,
            side=Side.LONG,
            symbol="EURUSD",
        )

        # Check it's a valid lot size
        assert size == round(size, 2)


class TestExposureLimits:
    """Tests for exposure limit checks."""

    def test_limits_exposure(self, risk_config):
        """Should reduce size when exposure limit reached."""
        risk_config.max_total_exposure_pct = 50.0
        sizer = PositionSizer(risk_config)

        adjusted_size, was_limited = sizer.check_exposure_limit(
            proposed_size=1.0,
            current_exposure=45000,  # 45% of 10000 equity
            equity=10000,
            entry_price=1.0800,
        )

        # Should be limited
        assert was_limited or adjusted_size < 1.0

    def test_allows_within_limit(self, risk_config):
        """Should allow size within limits."""
        risk_config.max_total_exposure_pct = 200.0  # 200% = $20,000 exposure allowed
        sizer = PositionSizer(risk_config)

        # 0.1 lot at 1.08 = $10,800 exposure, within 200% limit on $10,000 equity
        adjusted_size, was_limited = sizer.check_exposure_limit(
            proposed_size=0.1,
            current_exposure=0,
            equity=10000,
            entry_price=1.0800,
        )

        # Should not be limited since exposure < max
        assert abs(adjusted_size - 0.1) < 0.02
        assert not was_limited


class TestLeverageLimits:
    """Tests for leverage limit checks."""

    def test_limits_leverage(self, risk_config):
        """Should reduce size when leverage limit reached."""
        risk_config.max_leverage = 5.0
        sizer = PositionSizer(risk_config)

        adjusted_size, was_limited = sizer.check_leverage_limit(
            proposed_size=1.0,  # 100k notional
            current_exposure=40000,
            equity=10000,
            entry_price=1.0800,
        )

        # Current leverage would be ~14x, should be limited
        assert was_limited

    def test_allows_within_leverage(self, risk_config):
        """Should allow size within leverage limits."""
        risk_config.max_leverage = 10.0
        sizer = PositionSizer(risk_config)

        adjusted_size, was_limited = sizer.check_leverage_limit(
            proposed_size=0.1,  # 10k notional
            current_exposure=0,
            equity=10000,
            entry_price=1.0800,
        )

        assert not was_limited
