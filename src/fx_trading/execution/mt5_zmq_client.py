"""
ZeroMQ client for DWX MT5 bridge.

Implements the DWX protocol for communicating with MetaTrader 5 via ZeroMQ.
See: https://github.com/darwinex/dwxconnect
"""

import json
import time
from datetime import datetime
from typing import Any, Optional

import zmq
from loguru import logger


class MT5ZmqClient:
    """
    ZeroMQ client for DWX MT5 bridge.

    Communicates with the DWX_ZeroMQ_Server EA running in MetaTrader 5.

    Protocol:
    - PUSH socket sends commands to MT5
    - PULL socket receives responses from MT5
    - Commands are JSON-encoded strings
    """

    def __init__(
        self,
        host: str = "localhost",
        push_port: int = 32768,
        pull_port: int = 32769,
        timeout_ms: int = 30000,
    ):
        """
        Initialize ZeroMQ client.

        Args:
            host: MT5 machine hostname/IP
            push_port: Port for sending commands (PUSH socket)
            pull_port: Port for receiving responses (PULL socket)
            timeout_ms: Socket timeout in milliseconds
        """
        self.host = host
        self.push_port = push_port
        self.pull_port = pull_port
        self.timeout_ms = timeout_ms

        self.context: Optional[zmq.Context] = None
        self.push_socket: Optional[zmq.Socket] = None
        self.pull_socket: Optional[zmq.Socket] = None

        self._connected = False
        self._last_response: dict[str, Any] = {}

    def connect(self) -> bool:
        """
        Establish ZeroMQ connection to MT5.

        Returns:
            True if connection successful
        """
        try:
            self.context = zmq.Context()

            # PUSH socket for sending commands
            self.push_socket = self.context.socket(zmq.PUSH)
            self.push_socket.setsockopt(zmq.SNDTIMEO, self.timeout_ms)
            self.push_socket.connect(f"tcp://{self.host}:{self.push_port}")

            # PULL socket for receiving responses
            self.pull_socket = self.context.socket(zmq.PULL)
            self.pull_socket.setsockopt(zmq.RCVTIMEO, self.timeout_ms)
            self.pull_socket.connect(f"tcp://{self.host}:{self.pull_port}")

            self._connected = True
            logger.info(
                f"MT5 ZMQ connected: {self.host}:{self.push_port}/{self.pull_port}"
            )

            # Test connection with a ping
            account = self.get_account_info()
            if account:
                logger.info(f"MT5 account connected: {account.get('login', 'unknown')}")
                return True
            else:
                logger.warning("MT5 connection established but no account info received")
                return True  # Connection works, account might need login

        except zmq.ZMQError as e:
            logger.error(f"MT5 ZMQ connection failed: {e}")
            self._connected = False
            return False

    def disconnect(self) -> None:
        """Close ZeroMQ connection."""
        if self.push_socket:
            self.push_socket.close()
            self.push_socket = None

        if self.pull_socket:
            self.pull_socket.close()
            self.pull_socket = None

        if self.context:
            self.context.term()
            self.context = None

        self._connected = False
        logger.info("MT5 ZMQ disconnected")

    def is_connected(self) -> bool:
        """Check if connected to MT5."""
        return self._connected and self.push_socket is not None

    def _send_command(self, command: dict[str, Any]) -> bool:
        """
        Send command to MT5.

        Args:
            command: Command dictionary to send

        Returns:
            True if send successful
        """
        if not self.push_socket:
            logger.error("Cannot send: not connected")
            return False

        try:
            message = json.dumps(command)
            self.push_socket.send_string(message)
            logger.debug(f"Sent to MT5: {message}")
            return True
        except zmq.ZMQError as e:
            logger.error(f"Send failed: {e}")
            return False

    def _receive_response(self, timeout_ms: Optional[int] = None) -> Optional[dict[str, Any]]:
        """
        Receive response from MT5.

        Args:
            timeout_ms: Optional custom timeout

        Returns:
            Response dictionary or None if timeout/error
        """
        if not self.pull_socket:
            logger.error("Cannot receive: not connected")
            return None

        try:
            if timeout_ms:
                self.pull_socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

            message = self.pull_socket.recv_string()
            response = json.loads(message)
            logger.debug(f"Received from MT5: {message[:200]}...")
            self._last_response = response
            return response

        except zmq.Again:
            logger.warning("MT5 response timeout")
            return None
        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON from MT5: {e}")
            return None
        except zmq.ZMQError as e:
            logger.error(f"Receive failed: {e}")
            return None

    def _send_and_receive(
        self,
        command: dict[str, Any],
        timeout_ms: Optional[int] = None,
    ) -> Optional[dict[str, Any]]:
        """
        Send command and wait for response.

        Args:
            command: Command to send
            timeout_ms: Optional custom timeout

        Returns:
            Response dictionary or None
        """
        if not self._send_command(command):
            return None

        return self._receive_response(timeout_ms)

    # =========================================================================
    # DWX Protocol Commands
    # =========================================================================

    def get_tick(self, symbol: str) -> Optional[dict[str, Any]]:
        """
        Get current bid/ask for symbol.

        Args:
            symbol: Symbol name (e.g., "EURUSD")

        Returns:
            Dict with bid, ask, time or None
        """
        command = {
            "action": "GET_TICK",
            "symbol": symbol,
        }
        response = self._send_and_receive(command)

        if response and response.get("action") == "GET_TICK":
            return response.get("data", {})
        return None

    def get_account_info(self) -> Optional[dict[str, Any]]:
        """
        Get MT5 account information.

        Returns:
            Dict with balance, equity, margin, etc.
        """
        command = {"action": "GET_ACCOUNT_INFO"}
        response = self._send_and_receive(command)

        if response and response.get("action") == "GET_ACCOUNT_INFO":
            return response.get("data", {})
        return None

    def get_open_trades(self) -> Optional[dict[str, Any]]:
        """
        Get all open positions.

        Returns:
            Dict mapping ticket -> position info
        """
        command = {"action": "GET_OPEN_TRADES"}
        response = self._send_and_receive(command)

        if response and response.get("action") == "GET_OPEN_TRADES":
            return response.get("data", {})
        return None

    def open_trade(
        self,
        symbol: str,
        order_type: str,
        lots: float,
        price: float = 0.0,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
        magic: int = 0,
        comment: str = "",
        slippage: int = 10,
    ) -> Optional[dict[str, Any]]:
        """
        Open a new trade.

        Args:
            symbol: Symbol name
            order_type: "BUY" or "SELL" (or "BUY_LIMIT", "SELL_LIMIT", etc.)
            lots: Position size in lots
            price: Price for pending orders (0 for market)
            stop_loss: Stop loss price (0 for none)
            take_profit: Take profit price (0 for none)
            magic: Magic number for EA identification
            comment: Order comment
            slippage: Maximum slippage in points

        Returns:
            Dict with ticket number and execution details, or None
        """
        command = {
            "action": "OPEN_TRADE",
            "symbol": symbol,
            "type": order_type,
            "lots": lots,
            "price": price,
            "SL": stop_loss,
            "TP": take_profit,
            "magic": magic,
            "comment": comment,
            "slippage": slippage,
        }
        response = self._send_and_receive(command)

        if response:
            if response.get("action") == "OPEN_TRADE":
                return response.get("data", {})
            elif response.get("error"):
                logger.error(f"Trade open failed: {response.get('error')}")
                return None

        return None

    def close_trade(
        self,
        ticket: int,
        lots: float = 0.0,
        price: float = 0.0,
        slippage: int = 10,
    ) -> Optional[dict[str, Any]]:
        """
        Close an open position.

        Args:
            ticket: Position ticket number
            lots: Lots to close (0 = close all)
            price: Price for pending close (0 for market)
            slippage: Maximum slippage in points

        Returns:
            Dict with close details, or None
        """
        command = {
            "action": "CLOSE_TRADE",
            "ticket": ticket,
            "lots": lots,
            "price": price,
            "slippage": slippage,
        }
        response = self._send_and_receive(command)

        if response:
            if response.get("action") == "CLOSE_TRADE":
                return response.get("data", {})
            elif response.get("error"):
                logger.error(f"Trade close failed: {response.get('error')}")
                return None

        return None

    def close_all_trades(self) -> Optional[dict[str, Any]]:
        """
        Close all open positions.

        Returns:
            Dict with closed positions, or None
        """
        command = {"action": "CLOSE_ALL_TRADES"}
        response = self._send_and_receive(command)

        if response and response.get("action") == "CLOSE_ALL_TRADES":
            return response.get("data", {})
        return None

    def modify_trade(
        self,
        ticket: int,
        stop_loss: float = 0.0,
        take_profit: float = 0.0,
    ) -> Optional[dict[str, Any]]:
        """
        Modify stop loss / take profit for a position.

        Args:
            ticket: Position ticket number
            stop_loss: New stop loss price (0 to remove)
            take_profit: New take profit price (0 to remove)

        Returns:
            Dict with modification result, or None
        """
        command = {
            "action": "MODIFY_TRADE",
            "ticket": ticket,
            "SL": stop_loss,
            "TP": take_profit,
        }
        response = self._send_and_receive(command)

        if response:
            if response.get("action") == "MODIFY_TRADE":
                return response.get("data", {})
            elif response.get("error"):
                logger.error(f"Trade modify failed: {response.get('error')}")
                return None

        return None

    def get_historic_data(
        self,
        symbol: str,
        timeframe: str,
        start: datetime,
        end: datetime,
    ) -> Optional[list[dict[str, Any]]]:
        """
        Get historical OHLCV data.

        Args:
            symbol: Symbol name
            timeframe: Timeframe (e.g., "M1", "M5", "H1", "D1")
            start: Start datetime
            end: End datetime

        Returns:
            List of OHLCV bars, or None
        """
        command = {
            "action": "GET_HISTORIC_DATA",
            "symbol": symbol,
            "timeframe": timeframe,
            "start": start.strftime("%Y.%m.%d %H:%M"),
            "end": end.strftime("%Y.%m.%d %H:%M"),
        }
        response = self._send_and_receive(command, timeout_ms=60000)

        if response and response.get("action") == "GET_HISTORIC_DATA":
            return response.get("data", [])
        return None

    def subscribe_symbols(self, symbols: list[str]) -> bool:
        """
        Subscribe to symbol tick updates.

        Args:
            symbols: List of symbols to subscribe

        Returns:
            True if subscription successful
        """
        command = {
            "action": "SUBSCRIBE_SYMBOLS",
            "symbols": symbols,
        }
        response = self._send_and_receive(command)

        if response and response.get("action") == "SUBSCRIBE_SYMBOLS":
            logger.info(f"Subscribed to symbols: {symbols}")
            return True
        return False

    def get_last_error(self) -> Optional[str]:
        """Get last error from MT5."""
        return self._last_response.get("error")
