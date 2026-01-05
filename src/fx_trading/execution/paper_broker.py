"""
Paper Broker Implementation.

Simulates order execution for backtesting and paper trading.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID

from loguru import logger

from fx_trading.config.models import CostConfig
from fx_trading.costs.models import FillCalculator
from fx_trading.execution.broker import Broker
from fx_trading.portfolio.accounting import PortfolioManager
from fx_trading.types.models import (
    Order,
    OrderStatus,
    OrderType,
    Fill,
    Position,
    PositionStatus,
    AccountState,
    PriceData,
    Side,
)


class PaperBroker(Broker):
    """
    Paper broker for simulated trading.

    Features:
    - Realistic fill simulation with spread, slippage, commission
    - Stop loss and take profit management
    - Idempotency for order retries
    - Full position and order tracking
    """

    def __init__(
        self,
        portfolio: PortfolioManager,
        cost_config: CostConfig,
        seed: Optional[int] = None,
    ):
        """
        Initialize paper broker.

        Args:
            portfolio: Portfolio manager for position tracking
            cost_config: Cost configuration
            seed: Random seed for slippage
        """
        self.portfolio = portfolio
        self.cost_config = cost_config

        # Determine pip value (simplified)
        self.pip_value = 0.0001

        self.fill_calculator = FillCalculator(
            config=cost_config,
            pip_value=self.pip_value,
            seed=seed,
        )

        # Order tracking
        self.orders: dict[UUID, Order] = {}
        self.pending_orders: dict[UUID, Order] = {}
        self.processed_keys: set[str] = set()

        # Current prices (updated externally)
        self.current_prices: dict[str, PriceData] = {}

        # Current bar index for time-based exits
        self.current_bar_index: int = 0

        self._connected = False

        logger.info("Paper broker initialized")

    def connect(self) -> bool:
        """Connect (always succeeds for paper broker)."""
        self._connected = True
        logger.info("Paper broker connected")
        return True

    def disconnect(self) -> None:
        """Disconnect."""
        self._connected = False
        logger.info("Paper broker disconnected")

    def is_connected(self) -> bool:
        """Check connection status."""
        return self._connected

    def set_prices(self, prices: dict[str, PriceData]) -> None:
        """
        Update current prices.

        Args:
            prices: Dict of symbol -> PriceData
        """
        self.current_prices = prices

    def set_bar_index(self, index: int) -> None:
        """Set current bar index for time-based exits."""
        self.current_bar_index = index

    def get_prices(self, symbol: str) -> Optional[PriceData]:
        """Get current prices for symbol."""
        return self.current_prices.get(symbol)

    def place_order(self, order: Order) -> Optional[Fill]:
        """
        Place and immediately fill a market order.

        For paper trading, market orders are filled instantly.
        """
        # Check idempotency
        if self.check_idempotency(order.idempotency_key):
            logger.warning(f"Order already processed: {order.idempotency_key}")
            return None

        self.orders[order.id] = order

        # Get current prices
        price_data = self.current_prices.get(order.symbol)
        if price_data is None:
            logger.error(f"No price data for {order.symbol}")
            order.status = OrderStatus.REJECTED
            return None

        # Handle different order types
        if order.order_type == OrderType.MARKET:
            return self._fill_market_order(order, price_data)

        elif order.order_type in [OrderType.STOP_LOSS, OrderType.TAKE_PROFIT]:
            # These are attached to positions, process in check_pending_orders
            self.pending_orders[order.id] = order
            return None

        elif order.order_type == OrderType.LIMIT:
            self.pending_orders[order.id] = order
            return None

        return None

    def _fill_market_order(
        self,
        order: Order,
        price_data: PriceData,
    ) -> Fill:
        """Fill a market order."""
        # Calculate fill with costs
        fill = self.fill_calculator.calculate_fill(
            order=order,
            bid=price_data.bid,
            ask=price_data.ask,
            mid=price_data.mid,
            is_entry=True,
        )

        # Update order status
        order.status = OrderStatus.FILLED

        # Mark as processed
        self.processed_keys.add(order.idempotency_key)

        # Open position in portfolio
        position = self.portfolio.open_position(
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            entry_price=fill.fill_price,
            entry_time=fill.timestamp,
            entry_bar_index=self.current_bar_index,
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            time_exit_bars=order.time_in_force_bars,
            entry_commission=fill.commission,
            entry_slippage=fill.slippage,
        )

        # Store position ID in fill metadata
        fill.metadata["position_id"] = str(position.id)

        logger.info(
            f"Order filled: {order.symbol} {order.side.value} "
            f"size={order.size} @ {fill.fill_price:.5f}"
        )

        return fill

    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel a pending order."""
        if order_id in self.pending_orders:
            order = self.pending_orders.pop(order_id)
            order.status = OrderStatus.CANCELLED
            logger.info(f"Order cancelled: {order_id}")
            return True
        return False

    def get_order(self, order_id: UUID) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id) or self.pending_orders.get(order_id)

    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Get open positions."""
        return self.portfolio.get_open_positions(symbol)

    def get_position(self, position_id: UUID) -> Optional[Position]:
        """Get position by ID."""
        return self.portfolio.positions.get(position_id)

    def close_position(
        self,
        position_id: UUID,
        size: Optional[float] = None,
        exit_reason: str = "MANUAL",
    ) -> Optional[Fill]:
        """Close a position."""
        position = self.portfolio.positions.get(position_id)
        if position is None:
            logger.warning(f"Position not found: {position_id}")
            return None

        # Get exit price
        price_data = self.current_prices.get(position.symbol)
        if price_data is None:
            logger.error(f"No price data for {position.symbol}")
            return None

        # Create exit order
        exit_side = Side.SHORT if position.side == Side.LONG else Side.LONG
        exit_order = Order(
            symbol=position.symbol,
            side=exit_side,
            order_type=OrderType.MARKET,
            size=size or position.size,
            parent_position_id=position_id,
        )

        # Calculate exit fill
        fill = self.fill_calculator.calculate_fill(
            order=exit_order,
            bid=price_data.bid,
            ask=price_data.ask,
            mid=price_data.mid,
            is_entry=False,
        )

        # Close position in portfolio
        trade = self.portfolio.close_position(
            position_id=position_id,
            exit_price=fill.fill_price,
            exit_time=fill.timestamp,
            exit_bar_index=self.current_bar_index,
            exit_commission=fill.commission,
            exit_slippage=fill.slippage,
            exit_reason=exit_reason,
        )

        fill.metadata["trade_id"] = str(trade.id)
        fill.metadata["net_pnl"] = trade.net_pnl

        logger.info(
            f"Position closed: {position.symbol} {position.side.value} "
            f"pnl={trade.net_pnl:.2f} reason={exit_reason}"
        )

        return fill

    def close_all_positions(self) -> list[Fill]:
        """Close all open positions."""
        fills = []
        position_ids = list(self.portfolio.positions.keys())

        for position_id in position_ids:
            fill = self.close_position(position_id, exit_reason="CLOSE_ALL")
            if fill:
                fills.append(fill)

        logger.info(f"Closed {len(fills)} positions")
        return fills

    def get_account(self) -> AccountState:
        """Get current account state."""
        return self.portfolio.account

    def get_pending_orders(self) -> list[Order]:
        """Get all pending orders."""
        return list(self.pending_orders.values())

    def check_idempotency(self, idempotency_key: str) -> bool:
        """Check if order already processed."""
        return idempotency_key in self.processed_keys

    def check_pending_orders(self) -> list[Fill]:
        """
        Check and execute pending SL/TP orders.

        Should be called on each bar/tick.

        Returns:
            List of fills from triggered orders
        """
        fills = []
        triggered = []

        for order_id, order in self.pending_orders.items():
            if order.parent_position_id is None:
                continue

            position = self.portfolio.positions.get(order.parent_position_id)
            if position is None:
                triggered.append(order_id)
                continue

            price_data = self.current_prices.get(position.symbol)
            if price_data is None:
                continue

            # Check stop loss
            should_close = False
            exit_reason = ""

            if position.check_stop_loss(price_data.bid, price_data.ask):
                should_close = True
                exit_reason = "STOP_LOSS"

            elif position.check_take_profit(price_data.bid, price_data.ask):
                should_close = True
                exit_reason = "TAKE_PROFIT"

            if should_close:
                fill = self.close_position(
                    position.id,
                    exit_reason=exit_reason,
                )
                if fill:
                    fills.append(fill)
                triggered.append(order_id)

        # Remove triggered orders
        for order_id in triggered:
            self.pending_orders.pop(order_id, None)

        return fills

    def check_time_exits(self) -> list[Fill]:
        """
        Check and execute time-based exits.

        Returns:
            List of fills from time exits
        """
        fills = []
        position_ids = list(self.portfolio.positions.keys())

        for position_id in position_ids:
            position = self.portfolio.positions.get(position_id)
            if position is None:
                continue

            if position.check_time_exit(self.current_bar_index):
                fill = self.close_position(
                    position_id,
                    exit_reason="TIME_EXIT",
                )
                if fill:
                    fills.append(fill)

        return fills

    def update_positions(self) -> float:
        """
        Update all positions with current prices.

        Returns:
            Total unrealized PnL
        """
        prices = {}
        for symbol, price_data in self.current_prices.items():
            prices[symbol] = (price_data.bid, price_data.ask)

        return self.portfolio.update_positions(prices)
