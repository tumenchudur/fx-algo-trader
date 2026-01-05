"""
Cost models for realistic trade execution.

Implements spread, slippage, and commission modeling.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional

import numpy as np
from loguru import logger

from fx_trading.config.models import CostConfig
from fx_trading.types.models import Side, Fill, Order


@dataclass
class ExecutionCosts:
    """Breakdown of execution costs."""

    spread_cost: float = 0.0
    slippage_cost: float = 0.0
    commission: float = 0.0
    total: float = 0.0

    def __post_init__(self) -> None:
        """Calculate total."""
        self.total = self.spread_cost + self.slippage_cost + self.commission


class CostModel(ABC):
    """Abstract base for cost models."""

    @abstractmethod
    def calculate(self, price: float, size: float, side: Side) -> float:
        """Calculate cost component."""
        pass


class SpreadModel:
    """
    Models bid-ask spread.

    Spread affects entry and exit prices:
    - Long entry: pay ask (higher)
    - Long exit: receive bid (lower)
    - Short entry: receive bid
    - Short exit: pay ask
    """

    def __init__(
        self,
        use_provided_spread: bool = True,
        default_spread_pips: float = 1.5,
        spread_multiplier: float = 1.0,
        pip_value: float = 0.0001,
    ):
        """
        Initialize spread model.

        Args:
            use_provided_spread: Use bid/ask from data if available
            default_spread_pips: Default spread in pips
            spread_multiplier: Multiply spread (for volatile conditions)
            pip_value: Pip value (0.0001 for most pairs, 0.01 for JPY)
        """
        self.use_provided_spread = use_provided_spread
        self.default_spread_pips = default_spread_pips
        self.spread_multiplier = spread_multiplier
        self.pip_value = pip_value
        self.default_spread = default_spread_pips * pip_value

    def get_execution_price(
        self,
        side: Side,
        is_entry: bool,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        mid: Optional[float] = None,
    ) -> float:
        """
        Get execution price based on side and whether entry or exit.

        Args:
            side: Trade side (LONG or SHORT)
            is_entry: True for entry, False for exit
            bid: Bid price from data
            ask: Ask price from data
            mid: Mid price (used if bid/ask not available)

        Returns:
            Execution price
        """
        # If we have bid/ask, use them
        if self.use_provided_spread and bid is not None and ask is not None:
            # Apply spread multiplier
            if self.spread_multiplier > 1.0:
                spread = (ask - bid) * self.spread_multiplier
                mid_price = (bid + ask) / 2
                bid = mid_price - spread / 2
                ask = mid_price + spread / 2
        else:
            # Derive from mid price
            if mid is None:
                raise ValueError("Either bid/ask or mid price required")
            spread = self.default_spread * self.spread_multiplier
            bid = mid - spread / 2
            ask = mid + spread / 2

        # Determine execution price
        if side == Side.LONG:
            # Long: buy at ask, sell at bid
            return ask if is_entry else bid
        elif side == Side.SHORT:
            # Short: sell at bid, buy at ask
            return bid if is_entry else ask
        else:
            return mid if mid else (bid + ask) / 2

    def get_spread_cost(
        self,
        size: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        mid: Optional[float] = None,
    ) -> float:
        """
        Calculate spread cost for a round trip.

        Args:
            size: Position size in lots
            bid: Bid price
            ask: Ask price
            mid: Mid price

        Returns:
            Spread cost in account currency
        """
        if bid is not None and ask is not None:
            spread = (ask - bid) * self.spread_multiplier
        else:
            spread = self.default_spread * self.spread_multiplier

        # Cost = spread * lot_size * pip_value_per_lot
        # Standard lot = 100,000 units
        return spread * size * 100000


class SlippageModel:
    """
    Models execution slippage.

    Slippage occurs when actual fill price differs from expected price.
    Always works against the trader.
    """

    def __init__(
        self,
        model_type: str = "fixed_pips",
        slippage_pips: float = 0.5,
        slippage_percentage: float = 0.0001,
        volatility_multiplier: float = 0.1,
        pip_value: float = 0.0001,
        seed: Optional[int] = None,
    ):
        """
        Initialize slippage model.

        Args:
            model_type: "fixed_pips", "percentage", or "volatility_based"
            slippage_pips: Fixed slippage in pips
            slippage_percentage: Slippage as % of price
            volatility_multiplier: Multiplier for volatility-based slippage
            pip_value: Pip value
            seed: Random seed for reproducibility
        """
        self.model_type = model_type
        self.slippage_pips = slippage_pips
        self.slippage_percentage = slippage_percentage
        self.volatility_multiplier = volatility_multiplier
        self.pip_value = pip_value
        self.rng = np.random.default_rng(seed)

    def apply_slippage(
        self,
        price: float,
        side: Side,
        is_entry: bool,
        volatility: Optional[float] = None,
    ) -> tuple[float, float]:
        """
        Apply slippage to execution price.

        Slippage always works against the trader:
        - Long entry: price increases
        - Long exit: price decreases
        - Short entry: price decreases
        - Short exit: price increases

        Args:
            price: Base execution price (after spread)
            side: Trade side
            is_entry: True for entry, False for exit
            volatility: Optional ATR or volatility measure

        Returns:
            Tuple of (adjusted_price, slippage_amount)
        """
        # Calculate base slippage
        if self.model_type == "fixed_pips":
            base_slippage = self.slippage_pips * self.pip_value
        elif self.model_type == "percentage":
            base_slippage = price * self.slippage_percentage
        elif self.model_type == "volatility_based" and volatility is not None:
            base_slippage = volatility * self.volatility_multiplier
        else:
            base_slippage = self.slippage_pips * self.pip_value

        # Add randomness (0 to 2x base slippage)
        random_factor = self.rng.uniform(0.5, 1.5)
        slippage = base_slippage * random_factor

        # Direction: slippage always hurts
        if side == Side.LONG:
            # Long entry: pay more, Long exit: receive less
            direction = 1 if is_entry else -1
        elif side == Side.SHORT:
            # Short entry: receive less, Short exit: pay more
            direction = -1 if is_entry else 1
        else:
            return price, 0.0

        adjusted_price = price + (slippage * direction)
        return adjusted_price, slippage

    def get_slippage_cost(
        self,
        slippage_amount: float,
        size: float,
    ) -> float:
        """
        Calculate slippage cost in account currency.

        Args:
            slippage_amount: Slippage in price terms
            size: Position size in lots

        Returns:
            Slippage cost
        """
        return abs(slippage_amount) * size * 100000


class CommissionModel:
    """Models trading commissions."""

    def __init__(
        self,
        commission_type: str = "per_lot",
        commission_per_lot: float = 7.0,
        commission_percentage: float = 0.0,
    ):
        """
        Initialize commission model.

        Args:
            commission_type: "per_lot", "per_trade", or "percentage"
            commission_per_lot: Commission per standard lot
            commission_percentage: Commission as % of trade value
        """
        self.commission_type = commission_type
        self.commission_per_lot = commission_per_lot
        self.commission_percentage = commission_percentage

    def calculate(
        self,
        size: float,
        price: float,
    ) -> float:
        """
        Calculate commission for trade.

        Args:
            size: Position size in lots
            price: Execution price

        Returns:
            Commission amount
        """
        if self.commission_type == "per_lot":
            return self.commission_per_lot * size
        elif self.commission_type == "per_trade":
            return self.commission_per_lot
        elif self.commission_type == "percentage":
            trade_value = size * 100000 * price
            return trade_value * self.commission_percentage
        else:
            return self.commission_per_lot * size


class FillCalculator:
    """
    Calculates complete fill with all costs.

    Combines spread, slippage, and commission models.
    """

    def __init__(self, config: CostConfig, pip_value: float = 0.0001, seed: Optional[int] = None):
        """
        Initialize fill calculator.

        Args:
            config: Cost configuration
            pip_value: Pip value for the symbol
            seed: Random seed for reproducibility
        """
        self.config = config
        self.pip_value = pip_value

        self.spread_model = SpreadModel(
            use_provided_spread=config.use_provided_spread,
            default_spread_pips=config.default_spread_pips,
            spread_multiplier=config.spread_multiplier,
            pip_value=pip_value,
        )

        self.slippage_model = SlippageModel(
            model_type=config.slippage_model,
            slippage_pips=config.slippage_pips,
            slippage_percentage=config.slippage_percentage,
            volatility_multiplier=config.slippage_volatility_multiplier,
            pip_value=pip_value,
            seed=seed,
        )

        self.commission_model = CommissionModel(
            commission_type=config.commission_type,
            commission_per_lot=config.commission_per_lot,
            commission_percentage=config.commission_percentage,
        )

    def calculate_fill(
        self,
        order: Order,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        mid: Optional[float] = None,
        volatility: Optional[float] = None,
        is_entry: bool = True,
    ) -> Fill:
        """
        Calculate complete fill with all costs.

        Args:
            order: Order to fill
            bid: Current bid price
            ask: Current ask price
            mid: Current mid price
            volatility: Current volatility (for slippage)
            is_entry: True for entry, False for exit

        Returns:
            Fill with all costs calculated
        """
        from datetime import datetime
        from uuid import uuid4

        # Step 1: Get base price from spread model
        base_price = self.spread_model.get_execution_price(
            side=order.side,
            is_entry=is_entry,
            bid=bid,
            ask=ask,
            mid=mid,
        )

        # Step 2: Apply slippage
        fill_price, slippage_amount = self.slippage_model.apply_slippage(
            price=base_price,
            side=order.side,
            is_entry=is_entry,
            volatility=volatility,
        )

        # Step 3: Calculate commission
        commission = self.commission_model.calculate(
            size=order.size,
            price=fill_price,
        )

        # Calculate slippage in pips
        slippage_pips = slippage_amount / self.pip_value

        fill = Fill(
            id=uuid4(),
            order_id=order.id,
            timestamp=datetime.utcnow(),
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            fill_price=fill_price,
            commission=commission,
            slippage=slippage_amount,
            slippage_pips=slippage_pips,
            is_entry=is_entry,
            metadata={
                "bid": bid,
                "ask": ask,
                "base_price": base_price,
                "volatility": volatility,
            },
        )

        logger.debug(
            f"Fill calculated: {order.symbol} {order.side.value} "
            f"size={order.size} price={fill_price:.5f} "
            f"slippage={slippage_pips:.1f}pips commission={commission:.2f}"
        )

        return fill

    def get_total_costs(
        self,
        size: float,
        bid: Optional[float] = None,
        ask: Optional[float] = None,
        mid: Optional[float] = None,
    ) -> ExecutionCosts:
        """
        Estimate total round-trip costs.

        Args:
            size: Position size in lots
            bid: Bid price
            ask: Ask price
            mid: Mid price

        Returns:
            ExecutionCosts breakdown
        """
        spread_cost = self.spread_model.get_spread_cost(size, bid, ask, mid)

        # Estimate slippage (use expected value)
        avg_slippage = self.config.slippage_pips * self.pip_value
        slippage_cost = self.slippage_model.get_slippage_cost(avg_slippage, size) * 2  # Round trip

        # Commission for round trip
        price = mid or ((bid + ask) / 2 if bid and ask else 1.0)
        commission = self.commission_model.calculate(size, price) * 2  # Entry + exit

        return ExecutionCosts(
            spread_cost=spread_cost,
            slippage_cost=slippage_cost,
            commission=commission,
        )
