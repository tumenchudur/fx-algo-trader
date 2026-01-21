"""Tests for position management: trailing stop and partial take profit."""

import pytest
from uuid import uuid4

from fx_trading.trading.position_manager import (
    TrailingStopConfig,
    PartialTPConfig,
    TrailingStopManager,
    PartialTakeProfitManager,
    parse_trailing_stop_config,
    parse_partial_tp_config,
)
from fx_trading.types.models import Position, Side


class TestTrailingStopConfig:
    """Tests for trailing stop configuration parsing."""

    def test_parses_enabled_config(self):
        """Should parse enabled trailing stop config."""
        params = {
            "trailing_stop": {
                "enabled": True,
                "method": "atr_based",
                "atr_multiplier": 2.5,
                "activation_profit_atr": 1.5,
            }
        }
        config = parse_trailing_stop_config(params)
        assert config.enabled is True
        assert config.method == "atr_based"
        assert config.atr_multiplier == 2.5
        assert config.activation_profit_atr == 1.5

    def test_parses_disabled_config(self):
        """Should return disabled config when not specified."""
        config = parse_trailing_stop_config({})
        assert config.enabled is False

    def test_parses_fixed_pips_method(self):
        """Should parse fixed pips trailing config."""
        params = {
            "trailing_stop": {
                "enabled": True,
                "method": "fixed_pips",
                "fixed_pips": 50.0,
            }
        }
        config = parse_trailing_stop_config(params)
        assert config.method == "fixed_pips"
        assert config.fixed_pips == 50.0


class TestPartialTPConfig:
    """Tests for partial take profit configuration parsing."""

    def test_parses_enabled_config(self):
        """Should parse enabled partial TP config."""
        params = {
            "partial_tp": {
                "enabled": True,
                "first_target_r": 1.5,
                "first_close_pct": 40.0,
                "move_sl_to_breakeven": False,
            }
        }
        config = parse_partial_tp_config(params)
        assert config.enabled is True
        assert config.first_target_r == 1.5
        assert config.first_close_pct == 40.0
        assert config.move_sl_to_breakeven is False

    def test_parses_disabled_config(self):
        """Should return disabled config when not specified."""
        config = parse_partial_tp_config({})
        assert config.enabled is False

    def test_parses_second_target(self):
        """Should parse second target configuration."""
        params = {
            "partial_tp": {
                "enabled": True,
                "second_target_r": 2.0,
                "second_close_pct": 30.0,
            }
        }
        config = parse_partial_tp_config(params)
        assert config.second_target_r == 2.0
        assert config.second_close_pct == 30.0


class TestTrailingStopManager:
    """Tests for trailing stop manager."""

    @pytest.fixture
    def manager(self) -> TrailingStopManager:
        """Create trailing stop manager with ATR-based config."""
        config = TrailingStopConfig(
            enabled=True,
            method="atr_based",
            atr_multiplier=2.0,
            activation_profit_atr=1.0,
            step_pips=5.0,
        )
        return TrailingStopManager(config)

    @pytest.fixture
    def long_position(self) -> Position:
        """Create sample long position."""
        return Position(
            id=uuid4(),
            symbol="EURUSD",
            side=Side.LONG,
            size=0.1,
            entry_price=1.0800,
            stop_loss=1.0780,  # 20 pip SL
        )

    @pytest.fixture
    def short_position(self) -> Position:
        """Create sample short position."""
        return Position(
            id=uuid4(),
            symbol="EURUSD",
            side=Side.SHORT,
            size=0.1,
            entry_price=1.0800,
            stop_loss=1.0820,  # 20 pip SL
        )

    def test_no_trail_when_disabled(self, long_position):
        """Should not trail when disabled."""
        config = TrailingStopConfig(enabled=False)
        manager = TrailingStopManager(config)

        new_sl = manager.calculate_new_stop(long_position, 1.0900, atr=0.0010)
        assert new_sl is None

    def test_no_trail_without_atr_when_atr_based(self, manager, long_position):
        """Should not trail without ATR when method is atr_based."""
        new_sl = manager.calculate_new_stop(long_position, 1.0900, atr=None)
        assert new_sl is None

    def test_no_trail_insufficient_profit_long(self, manager, long_position):
        """Should not trail when profit is below activation threshold."""
        # With ATR=0.0010, activation = 1.0 * 0.0010 = 0.0010 (10 pips)
        # Entry=1.0800, need price >= 1.0810 to activate
        new_sl = manager.calculate_new_stop(long_position, 1.0805, atr=0.0010)
        assert new_sl is None

    def test_trails_when_profit_meets_threshold_long(self, manager, long_position):
        """Should trail when profit meets activation threshold."""
        # Price moved 15 pips up, above activation (10 pips)
        # Trail distance = 2.0 * 0.0010 = 0.0020
        # New SL = 1.0815 - 0.0020 = 1.0795
        new_sl = manager.calculate_new_stop(long_position, 1.0815, atr=0.0010)
        assert new_sl is not None
        assert new_sl > long_position.stop_loss

    def test_no_trail_insufficient_improvement(self, manager, long_position):
        """Should not trail if improvement is below minimum step."""
        # First trail
        first_new_sl = manager.calculate_new_stop(long_position, 1.0815, atr=0.0010)
        assert first_new_sl is not None

        # Update position's SL to the new value (simulating broker update)
        long_position.stop_loss = first_new_sl

        # Small price improvement - not enough to meet step_pips threshold
        new_sl = manager.calculate_new_stop(long_position, 1.0816, atr=0.0010)
        assert new_sl is None  # Improvement < step_pips (5 pips)

    def test_trails_short_position(self, manager, short_position):
        """Should trail short position correctly."""
        # Price moved down 15 pips
        # Trail distance = 2.0 * 0.0010 = 0.0020
        # New SL = 1.0785 + 0.0020 = 1.0805
        new_sl = manager.calculate_new_stop(short_position, 1.0785, atr=0.0010)
        assert new_sl is not None
        assert new_sl < short_position.stop_loss  # SL moved down for shorts

    def test_fixed_pips_trailing(self, long_position):
        """Should trail using fixed pips method."""
        config = TrailingStopConfig(
            enabled=True,
            method="fixed_pips",
            fixed_pips=30.0,  # 30 pip trail
        )
        manager = TrailingStopManager(config)

        # Price moved up significantly
        new_sl = manager.calculate_new_stop(long_position, 1.0850, atr=None)
        assert new_sl is not None
        # 1.0850 - 0.0030 = 1.0820
        assert abs(new_sl - 1.0820) < 0.0001


class TestPartialTakeProfitManager:
    """Tests for partial take profit manager."""

    @pytest.fixture
    def manager(self) -> PartialTakeProfitManager:
        """Create partial TP manager."""
        config = PartialTPConfig(
            enabled=True,
            first_target_r=1.0,
            first_close_pct=50.0,
            move_sl_to_breakeven=True,
        )
        return PartialTakeProfitManager(config)

    @pytest.fixture
    def long_position(self) -> Position:
        """Create sample long position."""
        return Position(
            id=uuid4(),
            symbol="EURUSD",
            side=Side.LONG,
            size=0.1,
            entry_price=1.0800,
            stop_loss=1.0780,  # 20 pip SL = 0.0020 initial risk
        )

    @pytest.fixture
    def short_position(self) -> Position:
        """Create sample short position."""
        return Position(
            id=uuid4(),
            symbol="EURUSD",
            side=Side.SHORT,
            size=0.1,
            entry_price=1.0800,
            stop_loss=1.0820,  # 20 pip SL
        )

    def test_no_partial_when_disabled(self, long_position):
        """Should not take partial when disabled."""
        config = PartialTPConfig(enabled=False)
        manager = PartialTakeProfitManager(config)

        action = manager.check_partial_exit(long_position, 1.0850)
        assert action is None

    def test_no_partial_below_target(self, manager, long_position):
        """Should not take partial below target."""
        # Entry=1.0800, SL=1.0780, risk=0.0020
        # 1R target = 1.0800 + 0.0020 = 1.0820
        action = manager.check_partial_exit(long_position, 1.0815)  # Below 1R
        assert action is None

    def test_partial_at_target(self, manager, long_position):
        """Should take partial at 1R target."""
        # 1R target = 1.0820
        action = manager.check_partial_exit(long_position, 1.0822)
        assert action is not None
        assert action.close_size == 0.05  # 50% of 0.1 lots
        assert action.new_sl == 1.0800  # Breakeven

    def test_only_one_first_partial(self, manager, long_position):
        """Should only take first partial once."""
        # First partial
        action1 = manager.check_partial_exit(long_position, 1.0825)
        assert action1 is not None

        # Second check at same level - no action
        action2 = manager.check_partial_exit(long_position, 1.0830)
        assert action2 is None

    def test_partial_short_position(self, manager, short_position):
        """Should handle short position partials."""
        # Entry=1.0800, SL=1.0820, risk=0.0020
        # 1R target = 1.0800 - 0.0020 = 1.0780
        action = manager.check_partial_exit(short_position, 1.0778)
        assert action is not None
        assert action.close_size == 0.05

    def test_no_breakeven_when_disabled(self, long_position):
        """Should not move SL to breakeven when disabled."""
        config = PartialTPConfig(
            enabled=True,
            first_target_r=1.0,
            first_close_pct=50.0,
            move_sl_to_breakeven=False,
        )
        manager = PartialTakeProfitManager(config)

        action = manager.check_partial_exit(long_position, 1.0822)
        assert action is not None
        assert action.new_sl is None

    def test_second_partial(self, long_position):
        """Should take second partial at configured target."""
        config = PartialTPConfig(
            enabled=True,
            first_target_r=1.0,
            first_close_pct=50.0,
            move_sl_to_breakeven=True,
            second_target_r=2.0,
            second_close_pct=25.0,
        )
        manager = PartialTakeProfitManager(config)

        # First partial at 1R
        action1 = manager.check_partial_exit(long_position, 1.0822)
        assert action1 is not None
        assert action1.close_size == 0.05  # 50% of 0.1

        # Second partial at 2R (entry + 2*risk = 1.0800 + 0.0040 = 1.0840)
        action2 = manager.check_partial_exit(long_position, 1.0842)
        assert action2 is not None
        # 25% of remaining 0.05 = 0.0125
        assert abs(action2.close_size - 0.0125) < 0.001

    def test_no_partial_without_stop_loss(self, manager):
        """Should not take partial without initial stop loss."""
        position = Position(
            id=uuid4(),
            symbol="EURUSD",
            side=Side.LONG,
            size=0.1,
            entry_price=1.0800,
            stop_loss=None,
        )
        action = manager.check_partial_exit(position, 1.0850)
        assert action is None


class TestGoldSymbolHandling:
    """Tests for XAUUSD/Gold symbol special handling."""

    def test_trailing_stop_gold_pip_value(self):
        """Should use correct pip value for gold (0.01)."""
        config = TrailingStopConfig(
            enabled=True,
            method="fixed_pips",
            fixed_pips=50.0,
        )
        manager = TrailingStopManager(config)

        position = Position(
            id=uuid4(),
            symbol="XAUUSD",
            side=Side.LONG,
            size=0.1,
            entry_price=2000.00,
            stop_loss=1990.00,
        )

        # Price moved up significantly
        new_sl = manager.calculate_new_stop(position, 2010.00, atr=None)
        assert new_sl is not None
        # 50 pips * 0.01 = $0.50 trail distance
        # 2010.00 - 0.50 = 2009.50
        assert abs(new_sl - 2009.50) < 0.01


class TestJPYSymbolHandling:
    """Tests for JPY pair special handling."""

    def test_trailing_stop_jpy_pip_value(self):
        """Should use correct pip value for JPY pairs (0.01)."""
        config = TrailingStopConfig(
            enabled=True,
            method="fixed_pips",
            fixed_pips=50.0,
        )
        manager = TrailingStopManager(config)

        position = Position(
            id=uuid4(),
            symbol="USDJPY",
            side=Side.LONG,
            size=0.1,
            entry_price=150.00,
            stop_loss=149.50,
        )

        # Price moved up significantly
        new_sl = manager.calculate_new_stop(position, 151.00, atr=None)
        assert new_sl is not None
        # 50 pips * 0.01 = 0.50 trail distance
        # 151.00 - 0.50 = 150.50
        assert abs(new_sl - 150.50) < 0.01
