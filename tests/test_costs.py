"""Tests for cost models."""

import pytest

from fx_trading.config.models import CostConfig
from fx_trading.costs.models import (
    SpreadModel,
    SlippageModel,
    CommissionModel,
    FillCalculator,
)
from fx_trading.types.models import Side, Order, OrderType


class TestSpreadModel:
    """Tests for spread model."""

    def test_long_entry_uses_ask(self):
        """Long entry should use ask price (higher)."""
        model = SpreadModel(use_provided_spread=True)
        price = model.get_execution_price(
            side=Side.LONG,
            is_entry=True,
            bid=1.0795,
            ask=1.0797,
        )
        assert price == 1.0797

    def test_long_exit_uses_bid(self):
        """Long exit should use bid price (lower)."""
        model = SpreadModel(use_provided_spread=True)
        price = model.get_execution_price(
            side=Side.LONG,
            is_entry=False,
            bid=1.0795,
            ask=1.0797,
        )
        assert price == 1.0795

    def test_short_entry_uses_bid(self):
        """Short entry should use bid price."""
        model = SpreadModel(use_provided_spread=True)
        price = model.get_execution_price(
            side=Side.SHORT,
            is_entry=True,
            bid=1.0795,
            ask=1.0797,
        )
        assert price == 1.0795

    def test_short_exit_uses_ask(self):
        """Short exit should use ask price (higher)."""
        model = SpreadModel(use_provided_spread=True)
        price = model.get_execution_price(
            side=Side.SHORT,
            is_entry=False,
            bid=1.0795,
            ask=1.0797,
        )
        assert price == 1.0797

    def test_derives_spread_from_mid(self):
        """Should derive bid/ask from mid when not provided."""
        model = SpreadModel(
            use_provided_spread=False,
            default_spread_pips=2.0,
        )
        price = model.get_execution_price(
            side=Side.LONG,
            is_entry=True,
            mid=1.0800,
        )
        # Ask = mid + spread/2 = 1.0800 + 0.0001 = 1.0801
        assert abs(price - 1.0801) < 0.00001

    def test_spread_cost_calculation(self):
        """Test spread cost calculation."""
        model = SpreadModel()
        cost = model.get_spread_cost(
            size=1.0,  # 1 lot
            bid=1.0795,
            ask=1.0797,  # 2 pip spread
        )
        # Cost = spread * size * 100000 = 0.0002 * 1.0 * 100000 = 20
        assert abs(cost - 20.0) < 0.01


class TestSlippageModel:
    """Tests for slippage model."""

    def test_slippage_hurts_long_entry(self):
        """Slippage should increase price on long entry."""
        model = SlippageModel(slippage_pips=1.0, seed=42)
        adjusted, slippage = model.apply_slippage(
            price=1.0800,
            side=Side.LONG,
            is_entry=True,
        )
        assert adjusted > 1.0800
        assert slippage > 0

    def test_slippage_hurts_long_exit(self):
        """Slippage should decrease price on long exit."""
        model = SlippageModel(slippage_pips=1.0, seed=42)
        adjusted, slippage = model.apply_slippage(
            price=1.0800,
            side=Side.LONG,
            is_entry=False,
        )
        assert adjusted < 1.0800
        assert slippage > 0

    def test_slippage_hurts_short_entry(self):
        """Slippage should decrease price on short entry."""
        model = SlippageModel(slippage_pips=1.0, seed=42)
        adjusted, slippage = model.apply_slippage(
            price=1.0800,
            side=Side.SHORT,
            is_entry=True,
        )
        assert adjusted < 1.0800

    def test_slippage_hurts_short_exit(self):
        """Slippage should increase price on short exit."""
        model = SlippageModel(slippage_pips=1.0, seed=42)
        adjusted, slippage = model.apply_slippage(
            price=1.0800,
            side=Side.SHORT,
            is_entry=False,
        )
        assert adjusted > 1.0800

    def test_reproducible_with_seed(self):
        """Same seed should produce same slippage."""
        model1 = SlippageModel(slippage_pips=1.0, seed=42)
        model2 = SlippageModel(slippage_pips=1.0, seed=42)

        result1 = model1.apply_slippage(1.0800, Side.LONG, True)
        result2 = model2.apply_slippage(1.0800, Side.LONG, True)

        assert result1[0] == result2[0]


class TestCommissionModel:
    """Tests for commission model."""

    def test_per_lot_commission(self):
        """Test per-lot commission calculation."""
        model = CommissionModel(
            commission_type="per_lot",
            commission_per_lot=7.0,
        )
        commission = model.calculate(size=2.5, price=1.0800)
        assert commission == 17.5  # 2.5 * 7

    def test_per_trade_commission(self):
        """Test per-trade commission."""
        model = CommissionModel(
            commission_type="per_trade",
            commission_per_lot=10.0,  # Used as per-trade fee
        )
        commission = model.calculate(size=5.0, price=1.0800)
        assert commission == 10.0  # Flat fee regardless of size

    def test_percentage_commission(self):
        """Test percentage-based commission."""
        model = CommissionModel(
            commission_type="percentage",
            commission_percentage=0.0001,  # 0.01%
        )
        commission = model.calculate(size=1.0, price=1.0800)
        # Trade value = 1.0 * 100000 * 1.08 = 108000
        # Commission = 108000 * 0.0001 = 10.8
        assert abs(commission - 10.8) < 0.01


class TestFillCalculator:
    """Tests for complete fill calculation."""

    def test_fill_includes_all_costs(self, cost_config):
        """Fill should include spread, slippage, and commission."""
        calc = FillCalculator(cost_config, seed=42)

        order = Order(
            symbol="EURUSD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            size=1.0,
        )

        fill = calc.calculate_fill(
            order=order,
            bid=1.0795,
            ask=1.0797,
            is_entry=True,
        )

        # Should enter at ask + slippage
        assert fill.fill_price >= 1.0797
        assert fill.commission > 0
        assert fill.slippage >= 0

    def test_fill_deterministic_with_seed(self, cost_config):
        """Fill should be deterministic with same seed."""
        calc1 = FillCalculator(cost_config, seed=42)
        calc2 = FillCalculator(cost_config, seed=42)

        order = Order(
            symbol="EURUSD",
            side=Side.LONG,
            order_type=OrderType.MARKET,
            size=1.0,
        )

        fill1 = calc1.calculate_fill(order, 1.0795, 1.0797, is_entry=True)
        fill2 = calc2.calculate_fill(order, 1.0795, 1.0797, is_entry=True)

        assert fill1.fill_price == fill2.fill_price
        assert fill1.slippage == fill2.slippage
