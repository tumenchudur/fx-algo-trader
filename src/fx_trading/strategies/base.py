"""
Base strategy interface and factory.

All strategies must implement the Strategy abstract class.
"""

from abc import ABC, abstractmethod
from typing import Optional, Type

import pandas as pd
from loguru import logger

from fx_trading.config.models import StrategyConfig
from fx_trading.types.models import Signal, Side


class Strategy(ABC):
    """
    Abstract base class for trading strategies.

    Strategies must be stateless or clearly manage their state.
    Signal generation must only use past/current data (no lookahead).
    """

    def __init__(self, config: StrategyConfig):
        """
        Initialize strategy.

        Args:
            config: Strategy configuration
        """
        self.config = config
        self.name = config.name
        self.symbols = config.symbols
        self.enabled = config.enabled

        # Extract strategy-specific params
        self.params = config.params

        logger.info(f"Strategy initialized: {self.name} ({config.strategy_type})")

    @abstractmethod
    def generate_signals(
        self,
        data: pd.DataFrame,
        current_index: int,
    ) -> list[Signal]:
        """
        Generate trading signals for current bar.

        CRITICAL: This method must only use data up to and including current_index.
        Accessing future data is lookahead bias and will corrupt backtest results.

        Args:
            data: Full OHLCV DataFrame (for calculating indicators)
            current_index: Current bar index (iloc position)

        Returns:
            List of Signal objects (empty if no signal)
        """
        pass

    def calculate_stop_loss(
        self,
        entry_price: float,
        side: Side,
        atr: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate stop loss price.

        Args:
            entry_price: Entry price
            side: Trade side
            atr: Optional ATR for volatility-based SL

        Returns:
            Stop loss price or None
        """
        if not self.config.use_stop_loss:
            return None

        # Default: 2 ATR or 1% of price
        if atr is not None:
            sl_distance = atr * self.params.get("sl_atr_multiplier", 2.0)
        else:
            sl_distance = entry_price * self.params.get("sl_pct", 0.01)

        if side == Side.LONG:
            return entry_price - sl_distance
        else:
            return entry_price + sl_distance

    def calculate_take_profit(
        self,
        entry_price: float,
        stop_loss: Optional[float],
        side: Side,
    ) -> Optional[float]:
        """
        Calculate take profit price based on risk-reward ratio.

        Args:
            entry_price: Entry price
            stop_loss: Stop loss price
            side: Trade side

        Returns:
            Take profit price or None
        """
        if not self.config.use_take_profit or stop_loss is None:
            return None

        sl_distance = abs(entry_price - stop_loss)
        tp_distance = sl_distance * self.config.risk_reward_ratio

        if side == Side.LONG:
            return entry_price + tp_distance
        else:
            return entry_price - tp_distance

    def get_lookback_period(self) -> int:
        """
        Get minimum bars needed for indicator calculation.

        Override in subclasses.
        """
        return self.params.get("lookback", 20)

    def validate_data(self, data: pd.DataFrame) -> bool:
        """
        Validate that data has required columns.

        Args:
            data: DataFrame to validate

        Returns:
            True if valid
        """
        required = ["open", "high", "low", "close"]
        missing = [c for c in required if c not in data.columns]

        if missing:
            logger.error(f"Missing required columns: {missing}")
            return False

        return True


class StrategyFactory:
    """Factory for creating strategy instances."""

    _strategies: dict[str, Type[Strategy]] = {}

    @classmethod
    def register(cls, name: str, strategy_class: Type[Strategy]) -> None:
        """Register a strategy type."""
        cls._strategies[name] = strategy_class
        logger.debug(f"Registered strategy: {name}")

    @classmethod
    def create(cls, config: StrategyConfig) -> Strategy:
        """
        Create strategy instance from config.

        Args:
            config: Strategy configuration

        Returns:
            Strategy instance

        Raises:
            ValueError: If strategy type not found
        """
        strategy_type = config.strategy_type

        if strategy_type not in cls._strategies:
            # Try importing standard strategies
            if strategy_type == "volatility_breakout":
                from fx_trading.strategies.volatility_breakout import VolatilityBreakoutStrategy
                cls.register("volatility_breakout", VolatilityBreakoutStrategy)
            elif strategy_type == "mean_reversion":
                from fx_trading.strategies.mean_reversion import MeanReversionStrategy
                cls.register("mean_reversion", MeanReversionStrategy)
            else:
                raise ValueError(f"Unknown strategy type: {strategy_type}")

        strategy_class = cls._strategies[strategy_type]
        return strategy_class(config)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """List registered strategy types."""
        return list(cls._strategies.keys())
