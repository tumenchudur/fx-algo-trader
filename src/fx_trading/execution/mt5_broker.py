"""
MetaTrader 5 Broker using the official Python package.

Implements the Broker interface for live/demo trading with MT5.
Requires the MetaTrader5 Python package and MT5 terminal running on Windows.
"""

from datetime import datetime
from typing import Optional
from uuid import UUID, uuid4

from loguru import logger

from fx_trading.execution.broker import Broker
from fx_trading.execution.mt5_client import MT5Client
from fx_trading.types.models import (
    Order,
    Fill,
    Position,
    AccountState,
    PriceData,
    Side,
    OrderType,
    OrderStatus,
    PositionStatus,
)


class MT5Broker(Broker):
    """
    MetaTrader 5 broker using official Python package.

    Translates between the internal trading system models and
    the MT5 Python API.
    """

    def __init__(
        self,
        client: MT5Client,
        symbol_suffix: str = "",
        magic_number: int = 123456,
    ):
        """
        Initialize MT5 broker.

        Args:
            client: MT5 client instance
            symbol_suffix: Suffix to add to symbols (e.g., "m" for micro)
            magic_number: Magic number for identifying our trades
        """
        self.client = client
        self.symbol_suffix = symbol_suffix
        self.magic_number = magic_number

        # Track our orders and positions by internal UUID
        self.orders: dict[UUID, Order] = {}
        self.pending_orders: dict[UUID, Order] = {}
        self.positions: dict[UUID, Position] = {}

        # Map internal UUIDs to MT5 ticket numbers
        self.ticket_to_uuid: dict[int, UUID] = {}
        self.uuid_to_ticket: dict[UUID, int] = {}

        # Idempotency tracking
        self.processed_keys: set[str] = set()

        # Account state cache
        self._account_state: Optional[AccountState] = None

    def connect(self) -> bool:
        """Connect to MT5."""
        if not self.client.connect():
            return False

        # Sync positions from MT5
        self._sync_positions()
        logger.info("MT5 broker connected and synced")
        return True

    def disconnect(self) -> None:
        """Disconnect from MT5."""
        self.client.disconnect()
        logger.info("MT5 broker disconnected")

    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        return self.client.is_connected()

    def _mt5_symbol(self, symbol: str) -> str:
        """Convert internal symbol to MT5 format."""
        return f"{symbol}{self.symbol_suffix}"

    def _internal_symbol(self, mt5_symbol: str) -> str:
        """Convert MT5 symbol to internal format."""
        if self.symbol_suffix and mt5_symbol.endswith(self.symbol_suffix):
            return mt5_symbol[: -len(self.symbol_suffix)]
        return mt5_symbol

    def get_prices(self, symbol: str) -> Optional[PriceData]:
        """Get current bid/ask from MT5."""
        mt5_symbol = self._mt5_symbol(symbol)
        tick = self.client.get_tick(mt5_symbol)

        if not tick:
            logger.warning(f"No tick data for {mt5_symbol}")
            return None

        try:
            return PriceData(
                timestamp=tick.get("time", datetime.utcnow()),
                symbol=symbol,
                bid=float(tick.get("bid", 0)),
                ask=float(tick.get("ask", 0)),
            )
        except (ValueError, KeyError) as e:
            logger.error(f"Invalid tick data: {e}")
            return None

    def place_order(self, order: Order) -> Optional[Fill]:
        """
        Place an order in MT5.

        For market orders, executes immediately and returns Fill.
        For pending orders, stores and returns None.
        """
        # Check idempotency
        if self.check_idempotency(order.idempotency_key):
            logger.warning(f"Order already processed: {order.idempotency_key}")
            return None

        # Store the order
        self.orders[order.id] = order

        # Convert order type
        if order.order_type == OrderType.MARKET:
            return self._execute_market_order(order)
        else:
            # Store pending order
            self.pending_orders[order.id] = order
            logger.info(f"Pending order created: {order.id}")
            return None

    def _execute_market_order(self, order: Order) -> Optional[Fill]:
        """Execute a market order in MT5."""
        mt5_symbol = self._mt5_symbol(order.symbol)

        # Determine order type
        if order.side == Side.LONG:
            order_type = "buy"
        elif order.side == Side.SHORT:
            order_type = "sell"
        else:
            logger.error(f"Invalid order side: {order.side}")
            return None

        # Place trade in MT5
        result = self.client.open_position(
            symbol=mt5_symbol,
            order_type=order_type,
            volume=order.size,
            sl=order.stop_loss or 0.0,
            tp=order.take_profit or 0.0,
            magic=self.magic_number,
            comment=f"FX_{order.id}",
        )

        if not result or not result.get("success"):
            order.status = OrderStatus.REJECTED
            logger.error(f"Order rejected: {order.id} - {result}")
            return None

        # Extract fill information
        ticket = result.get("ticket")
        fill_price = result.get("price", 0)

        if not ticket:
            logger.error(f"No ticket in MT5 response: {result}")
            order.status = OrderStatus.REJECTED
            return None

        # Track ticket mapping
        position_id = uuid4()
        self.ticket_to_uuid[ticket] = position_id
        self.uuid_to_ticket[position_id] = ticket

        # Create fill
        fill = Fill(
            order_id=order.id,
            timestamp=datetime.utcnow(),
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            fill_price=float(fill_price),
            commission=0,  # MT5 commission is per-deal, retrieved separately
            is_entry=True,
            metadata={
                "mt5_ticket": ticket,
                "position_id": str(position_id),
            },
        )

        # Create position
        position = Position(
            id=position_id,
            symbol=order.symbol,
            side=order.side,
            size=order.size,
            entry_price=float(fill_price),
            entry_time=datetime.utcnow(),
            current_price=float(fill_price),
            stop_loss=order.stop_loss,
            take_profit=order.take_profit,
            status=PositionStatus.OPEN,
        )
        self.positions[position_id] = position

        # Mark order as filled
        order.status = OrderStatus.FILLED
        self.processed_keys.add(order.idempotency_key)

        logger.info(
            f"Order filled: {order.symbol} {order.side.value} "
            f"size={order.size} @ {fill_price} (ticket={ticket})"
        )

        return fill

    def cancel_order(self, order_id: UUID) -> bool:
        """Cancel a pending order."""
        if order_id not in self.pending_orders:
            logger.warning(f"Order not found: {order_id}")
            return False

        order = self.pending_orders.pop(order_id)
        order.status = OrderStatus.CANCELLED

        # If we have a ticket, cancel in MT5
        if order_id in self.uuid_to_ticket:
            ticket = self.uuid_to_ticket[order_id]
            logger.info(f"Order cancelled: {order_id} (ticket={ticket})")
        else:
            logger.info(f"Order cancelled: {order_id}")

        return True

    def get_order(self, order_id: UUID) -> Optional[Order]:
        """Get order by ID."""
        return self.orders.get(order_id)

    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Get open positions."""
        positions = list(self.positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def get_position(self, position_id: UUID) -> Optional[Position]:
        """Get position by ID."""
        return self.positions.get(position_id)

    def close_position(
        self,
        position_id: UUID,
        size: Optional[float] = None,
    ) -> Optional[Fill]:
        """Close a position in MT5."""
        position = self.positions.get(position_id)
        if not position:
            logger.warning(f"Position not found: {position_id}")
            return None

        ticket = self.uuid_to_ticket.get(position_id)
        if not ticket:
            logger.error(f"No MT5 ticket for position: {position_id}")
            return None

        # Close the trade in MT5
        close_size = size or position.size
        result = self.client.close_position(
            ticket=ticket,
            volume=close_size,
        )

        if not result or not result.get("success"):
            # Check if position was already closed by MT5 (SL/TP hit)
            mt5_positions = self.client.get_open_positions()
            mt5_tickets = {p["ticket"] for p in mt5_positions} if mt5_positions else set()

            if ticket not in mt5_tickets:
                # Position was closed by MT5 (SL/TP), clean up our tracking
                logger.info(f"Position {position_id} was already closed by MT5 (SL/TP hit)")
                self._remove_position(position_id)
                return None

            logger.error(f"Failed to close position: {position_id} - {result}")
            return None

        # Get close price
        close_price = result.get("price", position.current_price)

        # Create fill for the close
        fill = Fill(
            order_id=uuid4(),
            timestamp=datetime.utcnow(),
            symbol=position.symbol,
            side=Side.SHORT if position.side == Side.LONG else Side.LONG,
            size=close_size,
            fill_price=float(close_price),
            commission=0,
            is_entry=False,
            metadata={
                "mt5_ticket": ticket,
                "position_id": str(position_id),
                "close_type": "manual",
            },
        )

        # Update position
        if size and size < position.size:
            # Partial close
            position.size -= size
            logger.info(f"Position partially closed: {position_id}, remaining={position.size}")
        else:
            # Full close
            position.status = PositionStatus.CLOSED
            del self.positions[position_id]
            logger.info(f"Position closed: {position_id} @ {close_price}")

        return fill

    def close_all_positions(self) -> list[Fill]:
        """Close all open positions."""
        fills = []

        # Close via MT5
        results = self.client.close_all_positions()
        if results:
            logger.info(f"All positions close requested: {len(results)} positions")

        # Update our tracking
        for position_id in list(self.positions.keys()):
            position = self.positions[position_id]
            position.status = PositionStatus.CLOSED

            fill = Fill(
                order_id=uuid4(),
                timestamp=datetime.utcnow(),
                symbol=position.symbol,
                side=Side.SHORT if position.side == Side.LONG else Side.LONG,
                size=position.size,
                fill_price=position.current_price,
                commission=0,
                is_entry=False,
                metadata={"close_type": "close_all"},
            )
            fills.append(fill)
            del self.positions[position_id]

        return fills

    def get_account(self) -> AccountState:
        """Get current account state from MT5."""
        info = self.client.get_account_info()

        if not info:
            # Return cached state if available
            if self._account_state:
                return self._account_state
            return AccountState()

        # Create account state from MT5 data
        balance = float(info.get("balance", 0))
        equity = float(info.get("equity", balance))
        margin_used = float(info.get("margin", 0))

        state = AccountState(
            timestamp=datetime.utcnow(),
            balance=balance,
            unrealized_pnl=equity - balance,
            margin_used=margin_used,
            leverage=float(info.get("leverage", 1)),
        )

        # Update peak/drawdown tracking
        if self._account_state:
            state.peak_equity = max(self._account_state.peak_equity, equity)
        else:
            state.peak_equity = equity

        state.update(state.unrealized_pnl, margin_used)

        self._account_state = state
        return state

    def get_pending_orders(self) -> list[Order]:
        """Get all pending orders."""
        return list(self.pending_orders.values())

    def check_idempotency(self, idempotency_key: str) -> bool:
        """Check if order was already processed."""
        return idempotency_key in self.processed_keys

    def _sync_positions(self) -> None:
        """Sync positions from MT5."""
        positions = self.client.get_open_positions()

        if not positions:
            logger.info("No open positions in MT5")
            return

        for pos in positions:
            try:
                ticket = pos["ticket"]

                # Skip if already tracked
                if ticket in self.ticket_to_uuid:
                    continue

                # Check if it's our trade (magic number)
                if pos.get("magic", 0) != self.magic_number:
                    logger.debug(f"Skipping foreign trade: {ticket}")
                    continue

                # Create position from MT5 data
                position_id = uuid4()
                symbol = self._internal_symbol(pos.get("symbol", ""))
                side = Side.LONG if pos.get("type") == "buy" else Side.SHORT

                position = Position(
                    id=position_id,
                    symbol=symbol,
                    side=side,
                    size=float(pos.get("volume", 0)),
                    entry_price=float(pos.get("price_open", 0)),
                    entry_time=pos.get("time", datetime.utcnow()),
                    current_price=float(pos.get("price_current", pos.get("price_open", 0))),
                    stop_loss=float(pos.get("sl", 0)) or None,
                    take_profit=float(pos.get("tp", 0)) or None,
                    status=PositionStatus.OPEN,
                )

                self.positions[position_id] = position
                self.ticket_to_uuid[ticket] = position_id
                self.uuid_to_ticket[position_id] = ticket

                logger.info(f"Synced position from MT5: {symbol} {side.value} (ticket={ticket})")

            except (ValueError, KeyError) as e:
                logger.error(f"Error syncing trade {pos}: {e}")

    def _remove_position(self, position_id: UUID) -> None:
        """Remove a position from internal tracking (when closed by MT5 SL/TP)."""
        if position_id in self.positions:
            del self.positions[position_id]

        if position_id in self.uuid_to_ticket:
            ticket = self.uuid_to_ticket.pop(position_id)
            if ticket in self.ticket_to_uuid:
                del self.ticket_to_uuid[ticket]

        logger.debug(f"Removed position from tracking: {position_id}")

    def update_positions(self) -> None:
        """Update position PnLs from current prices."""
        for position in self.positions.values():
            prices = self.get_prices(position.symbol)
            if prices:
                position.update_pnl(prices.bid, prices.ask)

    def modify_position_sl_tp(
        self,
        position_id: UUID,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """
        Modify stop loss / take profit for a position.

        Args:
            position_id: Internal position ID
            sl: New stop loss price (None = keep current)
            tp: New take profit price (None = keep current)

        Returns:
            True if modification successful
        """
        ticket = self.uuid_to_ticket.get(position_id)
        if not ticket:
            logger.error(f"No MT5 ticket for position: {position_id}")
            return False

        result = self.client.modify_position(ticket, sl, tp)
        if not result or not result.get("success"):
            logger.error(f"Failed to modify position: {position_id} - {result}")
            return False

        # Update local position
        position = self.positions.get(position_id)
        if position:
            if sl is not None:
                position.stop_loss = sl
            if tp is not None:
                position.take_profit = tp

        return True

    def reconcile_with_mt5(self) -> dict[str, list]:
        """
        Reconcile internal state with MT5.

        Returns:
            Dict with 'missing', 'extra', 'synced' position lists
        """
        result = {
            "missing": [],  # In MT5 but not tracked
            "extra": [],  # Tracked but not in MT5
            "synced": [],  # Matched
        }

        mt5_positions = self.client.get_open_positions()
        mt5_tickets = {p["ticket"] for p in mt5_positions}
        our_tickets = set(self.uuid_to_ticket.values())

        # Find missing (in MT5 but not tracked)
        for pos in mt5_positions:
            ticket = pos["ticket"]
            if ticket not in our_tickets and pos.get("magic") == self.magic_number:
                result["missing"].append(ticket)

        # Find extra (tracked but not in MT5)
        for ticket in our_tickets - mt5_tickets:
            result["extra"].append(ticket)

        # Find synced
        for ticket in mt5_tickets & our_tickets:
            result["synced"].append(ticket)

        if result["missing"]:
            logger.warning(f"Positions in MT5 not tracked: {result['missing']}")
        if result["extra"]:
            logger.warning(f"Tracked positions not in MT5: {result['extra']}")

        return result