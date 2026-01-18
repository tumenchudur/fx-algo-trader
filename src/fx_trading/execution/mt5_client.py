"""
MetaTrader 5 client using the official MetaTrader5 Python package.

This replaces the ZeroMQ-based DWX connector with the official MT5 Python API.
Requires MT5 terminal running on the same Windows machine.

See: https://www.mql5.com/en/docs/python_metatrader5
"""

from datetime import datetime
from typing import Any, Optional

from loguru import logger

try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None  # type: ignore
    logger.warning("MetaTrader5 package not installed. Install with: pip install MetaTrader5")


class MT5Client:
    """
    Client for MetaTrader 5 using the official Python package.

    This provides direct connection to MT5 without requiring an EA.
    MT5 terminal must be running on the same Windows machine.
    """

    def __init__(
        self,
        path: Optional[str] = None,
        login: Optional[int] = None,
        password: Optional[str] = None,
        server: Optional[str] = None,
        timeout: int = 60000,
        portable: bool = False,
    ):
        """
        Initialize MT5 client.

        Args:
            path: Path to MT5 terminal executable (optional, auto-detect)
            login: Account login number (optional, use terminal's default)
            password: Account password (optional, use terminal's saved)
            server: Broker server name (optional)
            timeout: Connection timeout in milliseconds
            portable: Use portable mode
        """
        self.path = path
        self.login = login
        self.password = password
        self.server = server
        self.timeout = timeout
        self.portable = portable
        self._connected = False

    def connect(self) -> bool:
        """
        Initialize connection to MT5 terminal.

        Returns:
            True if connection successful
        """
        if mt5 is None:
            logger.error("MetaTrader5 package not installed")
            return False

        # Build initialization kwargs
        init_kwargs: dict[str, Any] = {}
        if self.path:
            init_kwargs["path"] = self.path
        if self.timeout:
            init_kwargs["timeout"] = self.timeout
        if self.portable:
            init_kwargs["portable"] = self.portable

        # Initialize MT5 connection
        if not mt5.initialize(**init_kwargs):
            error = mt5.last_error()
            logger.error(f"MT5 initialization failed: {error}")
            return False

        # Login if credentials provided
        if self.login is not None:
            login_kwargs: dict[str, Any] = {"login": self.login}
            if self.password:
                login_kwargs["password"] = self.password
            if self.server:
                login_kwargs["server"] = self.server

            if not mt5.login(**login_kwargs):
                error = mt5.last_error()
                logger.error(f"MT5 login failed: {error}")
                mt5.shutdown()
                return False

        self._connected = True
        account_info = mt5.account_info()
        if account_info:
            logger.info(
                f"MT5 connected: {account_info.name} (#{account_info.login}) "
                f"on {account_info.server}"
            )
        return True

    def disconnect(self) -> None:
        """Shutdown MT5 connection."""
        if mt5 is not None and self._connected:
            mt5.shutdown()
            self._connected = False
            logger.info("MT5 disconnected")

    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        if not self._connected or mt5 is None:
            return False
        # Verify connection is still alive
        info = mt5.terminal_info()
        return info is not None and info.connected

    def get_account_info(self) -> Optional[dict[str, Any]]:
        """
        Get MT5 account information.

        Returns:
            Dict with balance, equity, margin, leverage, etc.
        """
        if not self.is_connected():
            return None

        info = mt5.account_info()
        if info is None:
            return None

        return {
            "login": info.login,
            "name": info.name,
            "server": info.server,
            "currency": info.currency,
            "balance": info.balance,
            "equity": info.equity,
            "margin": info.margin,
            "margin_free": info.margin_free,
            "margin_level": info.margin_level,
            "leverage": info.leverage,
            "profit": info.profit,
            "trade_mode": info.trade_mode,
        }

    def get_tick(self, symbol: str) -> Optional[dict[str, Any]]:
        """
        Get current bid/ask for symbol.

        Args:
            symbol: Symbol name (e.g., "EURUSD")

        Returns:
            Dict with bid, ask, time, etc.
        """
        if not self.is_connected():
            return None

        # Ensure symbol is selected in Market Watch
        if not mt5.symbol_select(symbol, True):
            logger.warning(f"Failed to select symbol: {symbol}")
            return None

        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None

        return {
            "symbol": symbol,
            "bid": tick.bid,
            "ask": tick.ask,
            "last": tick.last,
            "volume": tick.volume,
            "time": datetime.fromtimestamp(tick.time),
            "time_msc": tick.time_msc,
        }

    def get_symbol_info(self, symbol: str) -> Optional[dict[str, Any]]:
        """
        Get symbol specification.

        Args:
            symbol: Symbol name

        Returns:
            Dict with symbol properties (digits, lot sizes, etc.)
        """
        if not self.is_connected():
            return None

        info = mt5.symbol_info(symbol)
        if info is None:
            return None

        return {
            "name": info.name,
            "digits": info.digits,
            "point": info.point,
            "trade_tick_size": info.trade_tick_size,
            "trade_tick_value": info.trade_tick_value,
            "volume_min": info.volume_min,
            "volume_max": info.volume_max,
            "volume_step": info.volume_step,
            "trade_contract_size": info.trade_contract_size,
            "spread": info.spread,
            "bid": info.bid,
            "ask": info.ask,
        }

    def get_open_positions(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Get all open positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of position dicts
        """
        if not self.is_connected():
            return []

        if symbol:
            positions = mt5.positions_get(symbol=symbol)
        else:
            positions = mt5.positions_get()

        if positions is None:
            return []

        result = []
        for pos in positions:
            result.append({
                "ticket": pos.ticket,
                "symbol": pos.symbol,
                "type": "buy" if pos.type == mt5.POSITION_TYPE_BUY else "sell",
                "volume": pos.volume,
                "price_open": pos.price_open,
                "price_current": pos.price_current,
                "sl": pos.sl,
                "tp": pos.tp,
                "profit": pos.profit,
                "swap": pos.swap,
                "time": datetime.fromtimestamp(pos.time),
                "magic": pos.magic,
                "comment": pos.comment,
            })
        return result

    def get_pending_orders(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Get all pending orders.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of order dicts
        """
        if not self.is_connected():
            return []

        if symbol:
            orders = mt5.orders_get(symbol=symbol)
        else:
            orders = mt5.orders_get()

        if orders is None:
            return []

        order_types = {
            mt5.ORDER_TYPE_BUY_LIMIT: "buy_limit",
            mt5.ORDER_TYPE_SELL_LIMIT: "sell_limit",
            mt5.ORDER_TYPE_BUY_STOP: "buy_stop",
            mt5.ORDER_TYPE_SELL_STOP: "sell_stop",
        }

        result = []
        for order in orders:
            result.append({
                "ticket": order.ticket,
                "symbol": order.symbol,
                "type": order_types.get(order.type, "unknown"),
                "volume": order.volume_current,
                "price_open": order.price_open,
                "sl": order.sl,
                "tp": order.tp,
                "time_setup": datetime.fromtimestamp(order.time_setup),
                "magic": order.magic,
                "comment": order.comment,
            })
        return result

    def open_position(
        self,
        symbol: str,
        order_type: str,
        volume: float,
        price: Optional[float] = None,
        sl: float = 0.0,
        tp: float = 0.0,
        magic: int = 0,
        comment: str = "",
        deviation: int = 20,
    ) -> Optional[dict[str, Any]]:
        """
        Open a new position (market order).

        Args:
            symbol: Symbol name
            order_type: "buy" or "sell"
            volume: Position size in lots
            price: Optional price (uses market if None)
            sl: Stop loss price (0 = none)
            tp: Take profit price (0 = none)
            magic: Magic number for EA identification
            comment: Order comment
            deviation: Maximum price deviation in points

        Returns:
            Dict with order result, or None on failure
        """
        if not self.is_connected():
            return None

        # Get symbol info for price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None

        # Determine order type and price
        if order_type.lower() == "buy":
            mt5_type = mt5.ORDER_TYPE_BUY
            exec_price = price if price else tick.ask
        elif order_type.lower() == "sell":
            mt5_type = mt5.ORDER_TYPE_SELL
            exec_price = price if price else tick.bid
        else:
            logger.error(f"Invalid order type: {order_type}")
            return None

        # Get symbol info to determine correct filling mode
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            logger.error(f"Symbol info not found: {symbol}")
            return None

        # Use FOK filling mode (Fill or Kill) - try this first for JustMarkets
        filling_mode = mt5.ORDER_FILLING_FOK if hasattr(mt5, 'ORDER_FILLING_FOK') else 0

        # Build minimal request - only required fields
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": mt5_type,
            "price": float(exec_price),
            "deviation": int(deviation),
            "magic": int(magic),
            "type_filling": filling_mode,
        }

        # Add SL/TP only if set (some brokers don't allow SL/TP on market orders)
        if sl > 0:
            request["sl"] = float(sl)
        if tp > 0:
            request["tp"] = float(tp)

        logger.debug(f"Order request: {request}")

        # Send order directly (skip order_check - it has bugs)
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Order send failed: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Order failed: {result.retcode} - {result.comment}")
            return {
                "success": False,
                "retcode": result.retcode,
                "comment": result.comment,
            }

        logger.info(
            f"Order executed: {symbol} {order_type} {volume} lots @ {result.price} "
            f"(ticket={result.order})"
        )

        return {
            "success": True,
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
            "retcode": result.retcode,
            "comment": result.comment,
        }

    def close_position(
        self,
        ticket: int,
        volume: Optional[float] = None,
        deviation: int = 20,
    ) -> Optional[dict[str, Any]]:
        """
        Close an open position.

        Args:
            ticket: Position ticket number
            volume: Volume to close (None = close all)
            deviation: Maximum price deviation in points

        Returns:
            Dict with close result, or None on failure
        """
        if not self.is_connected():
            return None

        # Get position
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error(f"Position not found: {ticket}")
            return None

        position = position[0]
        symbol = position.symbol
        close_volume = volume if volume else position.volume

        # Get current price
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            logger.error(f"Failed to get tick for {symbol}")
            return None

        # Determine close type and price
        if position.type == mt5.POSITION_TYPE_BUY:
            close_type = mt5.ORDER_TYPE_SELL
            price = tick.bid
        else:
            close_type = mt5.ORDER_TYPE_BUY
            price = tick.ask

        # Build request
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": close_volume,
            "type": close_type,
            "position": ticket,
            "price": price,
            "deviation": deviation,
            "magic": position.magic,
            "comment": "close",
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        # Send order
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Close order failed: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Close failed: {result.retcode} - {result.comment}")
            return {
                "success": False,
                "retcode": result.retcode,
                "comment": result.comment,
            }

        logger.info(f"Position closed: {ticket} @ {result.price}")

        return {
            "success": True,
            "ticket": result.order,
            "price": result.price,
            "volume": result.volume,
            "retcode": result.retcode,
            "comment": result.comment,
        }

    def close_all_positions(self, symbol: Optional[str] = None) -> list[dict[str, Any]]:
        """
        Close all open positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of close results
        """
        positions = self.get_open_positions(symbol)
        results = []

        for pos in positions:
            result = self.close_position(pos["ticket"])
            if result:
                results.append(result)

        return results

    def modify_position(
        self,
        ticket: int,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Modify stop loss / take profit for a position.

        Args:
            ticket: Position ticket number
            sl: New stop loss price (None = keep current)
            tp: New take profit price (None = keep current)

        Returns:
            Dict with modification result, or None on failure
        """
        if not self.is_connected():
            return None

        # Get position
        position = mt5.positions_get(ticket=ticket)
        if not position:
            logger.error(f"Position not found: {ticket}")
            return None

        position = position[0]

        # Use current values if not specified
        new_sl = sl if sl is not None else position.sl
        new_tp = tp if tp is not None else position.tp

        # Build request
        request = {
            "action": mt5.TRADE_ACTION_SLTP,
            "symbol": position.symbol,
            "position": ticket,
            "sl": new_sl,
            "tp": new_tp,
        }

        # Send modification
        result = mt5.order_send(request)
        if result is None:
            logger.error(f"Modify failed: {mt5.last_error()}")
            return None

        if result.retcode != mt5.TRADE_RETCODE_DONE:
            logger.error(f"Modify failed: {result.retcode} - {result.comment}")
            return {
                "success": False,
                "retcode": result.retcode,
                "comment": result.comment,
            }

        logger.info(f"Position modified: {ticket} SL={new_sl} TP={new_tp}")

        return {
            "success": True,
            "retcode": result.retcode,
            "comment": result.comment,
        }

    def get_history_deals(
        self,
        from_date: datetime,
        to_date: Optional[datetime] = None,
    ) -> list[dict[str, Any]]:
        """
        Get historical deals.

        Args:
            from_date: Start date
            to_date: End date (default: now)

        Returns:
            List of deal dicts
        """
        if not self.is_connected():
            return []

        if to_date is None:
            to_date = datetime.now()

        deals = mt5.history_deals_get(from_date, to_date)
        if deals is None:
            return []

        deal_types = {
            mt5.DEAL_TYPE_BUY: "buy",
            mt5.DEAL_TYPE_SELL: "sell",
        }

        result = []
        for deal in deals:
            result.append({
                "ticket": deal.ticket,
                "order": deal.order,
                "position_id": deal.position_id,
                "symbol": deal.symbol,
                "type": deal_types.get(deal.type, "other"),
                "volume": deal.volume,
                "price": deal.price,
                "profit": deal.profit,
                "commission": deal.commission,
                "swap": deal.swap,
                "time": datetime.fromtimestamp(deal.time),
                "magic": deal.magic,
                "comment": deal.comment,
            })
        return result

    def get_last_error(self) -> tuple[int, str]:
        """Get last error code and description."""
        if mt5 is None:
            return (-1, "MetaTrader5 package not installed")
        return mt5.last_error()