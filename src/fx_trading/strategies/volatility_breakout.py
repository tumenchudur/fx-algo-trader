"""
Volatility Breakout Strategy.

Trades breakouts of recent highs/lows when volatility expands.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from fx_trading.config.models import StrategyConfig
from fx_trading.strategies.base import Strategy
from fx_trading.types.models import Signal, Side


class VolatilityBreakoutStrategy(Strategy):
    """
    Volatility Breakout Strategy.

    Entry Logic:
    - Long: Price breaks above the high of the lookback period
            AND ATR is above average (volatility expanding)
    - Short: Price breaks below the low of the lookback period
             AND ATR is above average

    Exit Logic:
    - Stop loss at entry minus/plus N*ATR
    - Take profit at risk-reward ratio
    - Optional time-based exit

    Parameters (via config.params):
    - lookback: Period for high/low calculation (default: 20)
    - atr_period: ATR calculation period (default: 14)
    - atr_threshold: ATR multiple above average to trade (default: 1.0)
    - sl_atr_multiplier: ATR multiplier for stop loss (default: 2.0)
    """

    def __init__(self, config: StrategyConfig):
        """Initialize with config."""
        super().__init__(config)

        # Extract parameters with defaults
        self.lookback = self.params.get("lookback", 20)
        self.atr_period = self.params.get("atr_period", 14)
        self.atr_threshold = self.params.get("atr_threshold", 1.0)
        self.sl_atr_multiplier = self.params.get("sl_atr_multiplier", 2.0)
        self.require_close_above = self.params.get("require_close_above", True)

        logger.info(
            f"VolatilityBreakout params: lookback={self.lookback}, "
            f"atr_period={self.atr_period}, atr_threshold={self.atr_threshold}"
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_index: int,
    ) -> list[Signal]:
        """
        Generate breakout signals.

        Uses only data up to current_index (no lookahead).
        """
        signals = []

        if not self.enabled:
            return signals

        if not self.validate_data(data):
            return signals

        # Need enough data for indicators
        min_required = max(self.lookback, self.atr_period * 2) + 1
        if current_index < min_required:
            return signals

        # Get current bar and historical data (NO LOOKAHEAD)
        current_bar = data.iloc[current_index]
        historical = data.iloc[:current_index + 1]

        # Calculate ATR indicators using only past data
        atr, atr_series = self._calculate_atr(historical)
        avg_atr = atr_series.rolling(self.atr_period * 2).mean().iloc[-1]

        if pd.isna(atr) or pd.isna(avg_atr):
            return signals

        # Check volatility condition
        volatility_expanding = atr >= avg_atr * self.atr_threshold

        if not volatility_expanding:
            return signals

        # Calculate breakout levels (excluding current bar)
        lookback_data = data.iloc[current_index - self.lookback:current_index]
        upper_level = lookback_data["high"].max()
        lower_level = lookback_data["low"].min()

        # Get current price
        current_high = current_bar["high"]
        current_low = current_bar["low"]
        current_close = current_bar["close"]

        # Determine signal
        signal_side = None

        # Long breakout
        if current_high > upper_level:
            if self.require_close_above:
                if current_close > upper_level:
                    signal_side = Side.LONG
            else:
                signal_side = Side.LONG

        # Short breakout
        elif current_low < lower_level:
            if self.require_close_above:
                if current_close < lower_level:
                    signal_side = Side.SHORT
            else:
                signal_side = Side.SHORT

        if signal_side is not None:
            # Calculate entry price (use close for signal, actual entry uses bid/ask)
            entry_price = current_close

            # Calculate stop loss
            sl_distance = atr * self.sl_atr_multiplier
            if signal_side == Side.LONG:
                stop_loss = entry_price - sl_distance
            else:
                stop_loss = entry_price + sl_distance

            # Calculate take profit
            take_profit = self.calculate_take_profit(entry_price, stop_loss, signal_side)

            # Get symbol from data if available
            symbol = current_bar.get("symbol", self.symbols[0]) if hasattr(current_bar, "get") else self.symbols[0]
            if isinstance(symbol, pd.Series):
                symbol = symbol.iloc[0] if len(symbol) > 0 else self.symbols[0]

            signal = Signal(
                timestamp=data.index[current_index] if isinstance(data.index[current_index], datetime) else datetime.utcnow(),
                symbol=symbol,
                side=signal_side,
                strength=min(1.0, atr / avg_atr / 2),  # Strength based on volatility
                stop_loss=stop_loss,
                take_profit=take_profit,
                time_exit_bars=self.config.time_exit_bars,
                metadata={
                    "upper_level": upper_level,
                    "lower_level": lower_level,
                    "atr": atr,
                    "avg_atr": avg_atr,
                },
            )
            signals.append(signal)

            logger.debug(
                f"Breakout signal: {signal_side.value} @ {entry_price:.5f}, "
                f"SL={stop_loss:.5f}, TP={take_profit}"
            )

        return signals

    def _calculate_atr(self, data: pd.DataFrame) -> tuple[float, pd.Series]:
        """
        Calculate ATR (Average True Range).

        Args:
            data: Historical OHLC data

        Returns:
            Tuple of (current ATR value, ATR series)
        """
        # Calculate True Range
        high = data["high"]
        low = data["low"]
        close = data["close"].shift(1)

        tr1 = high - low
        tr2 = abs(high - close)
        tr3 = abs(low - close)

        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Calculate ATR series
        atr_series = tr.rolling(self.atr_period).mean()

        return atr_series.iloc[-1], atr_series

    def get_lookback_period(self) -> int:
        """Get minimum lookback needed."""
        return max(self.lookback, self.atr_period * 2) + 1
