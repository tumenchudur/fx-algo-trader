"""
Abstract Broker Interface.

Defines the contract for all broker implementations.
"""

from abc import ABC, abstractmethod
from typing import Optional
from uuid import UUID

from fx_trading.types.models import (
    Order,
    Fill,
    Position,
    AccountState,
    PriceData,
)


class Broker(ABC):
    """
    Abstract broker interface.

    All broker implementations (paper, live) must implement this interface.
    This allows the backtester and paper trader to work with any broker.
    """

    @abstractmethod
    def connect(self) -> bool:
        """
        Connect to broker.

        Returns:
            True if connection successful
        """
        pass

    @abstractmethod
    def disconnect(self) -> None:
        """Disconnect from broker."""
        pass

    @abstractmethod
    def is_connected(self) -> bool:
        """Check if connected to broker."""
        pass

    @abstractmethod
    def get_prices(self, symbol: str) -> Optional[PriceData]:
        """
        Get current prices for symbol.

        Args:
            symbol: Trading symbol

        Returns:
            PriceData with bid/ask or None if unavailable
        """
        pass

    @abstractmethod
    def place_order(self, order: Order) -> Optional[Fill]:
        """
        Place an order.

        Args:
            order: Order to place

        Returns:
            Fill if executed, None if pending/failed
        """
        pass

    @abstractmethod
    def cancel_order(self, order_id: UUID) -> bool:
        """
        Cancel a pending order.

        Args:
            order_id: Order UUID

        Returns:
            True if cancelled successfully
        """
        pass

    @abstractmethod
    def get_order(self, order_id: UUID) -> Optional[Order]:
        """
        Get order by ID.

        Args:
            order_id: Order UUID

        Returns:
            Order or None
        """
        pass

    @abstractmethod
    def get_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """
        Get open positions.

        Args:
            symbol: Optional symbol filter

        Returns:
            List of open positions
        """
        pass

    @abstractmethod
    def get_position(self, position_id: UUID) -> Optional[Position]:
        """
        Get position by ID.

        Args:
            position_id: Position UUID

        Returns:
            Position or None
        """
        pass

    @abstractmethod
    def close_position(
        self,
        position_id: UUID,
        size: Optional[float] = None,
    ) -> Optional[Fill]:
        """
        Close a position.

        Args:
            position_id: Position UUID
            size: Partial close size (None = full close)

        Returns:
            Fill if closed, None if failed
        """
        pass

    @abstractmethod
    def close_all_positions(self) -> list[Fill]:
        """
        Close all open positions.

        Returns:
            List of fills
        """
        pass

    @abstractmethod
    def get_account(self) -> AccountState:
        """
        Get current account state.

        Returns:
            AccountState with balance, equity, etc.
        """
        pass

    @abstractmethod
    def get_pending_orders(self) -> list[Order]:
        """
        Get all pending orders.

        Returns:
            List of pending orders
        """
        pass

    def check_idempotency(self, idempotency_key: str) -> bool:
        """
        Check if order with key has already been processed.

        Prevents double execution on retries.

        Args:
            idempotency_key: Unique order key

        Returns:
            True if already processed
        """
        # Default implementation - override in subclasses
        return False
