"""
Mean Reversion Strategy.

Trades reversions to the mean when price deviates significantly.
"""

from datetime import datetime
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from fx_trading.config.models import StrategyConfig
from fx_trading.strategies.base import Strategy
from fx_trading.types.models import Signal, Side


class MeanReversionStrategy(Strategy):
    """
    Mean Reversion Strategy.

    Entry Logic:
    - Long: Z-score of price vs MA falls below -threshold (oversold)
    - Short: Z-score of price vs MA rises above +threshold (overbought)

    Exit Logic:
    - Exit when price returns to mean (z-score crosses zero)
    - Stop loss at entry minus/plus N*std
    - Take profit at mean or risk-reward ratio

    Parameters (via config.params):
    - lookback: Period for moving average (default: 20)
    - z_threshold: Z-score threshold for entry (default: 2.0)
    - sl_std_multiplier: Std multiplier for stop loss (default: 3.0)
    - exit_at_mean: Exit when price returns to mean (default: True)
    """

    def __init__(self, config: StrategyConfig):
        """Initialize with config."""
        super().__init__(config)

        # Extract parameters with defaults
        self.lookback = self.params.get("lookback", 20)
        self.z_threshold = self.params.get("z_threshold", 2.0)
        self.sl_std_multiplier = self.params.get("sl_std_multiplier", 3.0)
        self.exit_at_mean = self.params.get("exit_at_mean", True)
        self.use_returns = self.params.get("use_returns", False)  # Z-score of returns vs price

        logger.info(
            f"MeanReversion params: lookback={self.lookback}, "
            f"z_threshold={self.z_threshold}, use_returns={self.use_returns}"
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_index: int,
    ) -> list[Signal]:
        """
        Generate mean reversion signals.

        Uses only data up to current_index (no lookahead).
        """
        signals = []

        if not self.enabled:
            return signals

        if not self.validate_data(data):
            return signals

        # Need enough data for indicators
        min_required = self.lookback + 1
        if current_index < min_required:
            return signals

        # Get current bar and historical data (NO LOOKAHEAD)
        current_bar = data.iloc[current_index]
        historical = data.iloc[:current_index + 1]

        # Calculate z-score
        z_score, ma, std = self._calculate_zscore(historical)

        if pd.isna(z_score) or pd.isna(ma) or pd.isna(std) or std == 0:
            return signals

        # Get previous z-score for crossover detection
        if current_index > min_required:
            prev_historical = data.iloc[:current_index]
            prev_z, _, _ = self._calculate_zscore(prev_historical)
        else:
            prev_z = z_score

        # Determine signal
        signal_side = None
        current_close = current_bar["close"]

        # Long signal: z-score crosses below -threshold (oversold)
        if z_score < -self.z_threshold and prev_z >= -self.z_threshold:
            signal_side = Side.LONG

        # Short signal: z-score crosses above +threshold (overbought)
        elif z_score > self.z_threshold and prev_z <= self.z_threshold:
            signal_side = Side.SHORT

        if signal_side is not None:
            entry_price = current_close

            # Calculate stop loss
            sl_distance = std * self.sl_std_multiplier
            if signal_side == Side.LONG:
                stop_loss = entry_price - sl_distance
                # Take profit at mean or RR ratio
                if self.exit_at_mean:
                    take_profit = ma
                else:
                    take_profit = self.calculate_take_profit(entry_price, stop_loss, signal_side)
            else:
                stop_loss = entry_price + sl_distance
                if self.exit_at_mean:
                    take_profit = ma
                else:
                    take_profit = self.calculate_take_profit(entry_price, stop_loss, signal_side)

            # Get symbol from data if available
            symbol = current_bar.get("symbol", self.symbols[0]) if hasattr(current_bar, "get") else self.symbols[0]
            if isinstance(symbol, pd.Series):
                symbol = symbol.iloc[0] if len(symbol) > 0 else self.symbols[0]

            # Signal strength based on how extreme the z-score is
            strength = min(1.0, abs(z_score) / (self.z_threshold * 2))

            signal = Signal(
                timestamp=data.index[current_index] if isinstance(data.index[current_index], datetime) else datetime.utcnow(),
                symbol=symbol,
                side=signal_side,
                strength=strength,
                stop_loss=stop_loss,
                take_profit=take_profit,
                time_exit_bars=self.config.time_exit_bars,
                metadata={
                    "z_score": z_score,
                    "ma": ma,
                    "std": std,
                    "deviation": current_close - ma,
                },
            )
            signals.append(signal)

            logger.debug(
                f"Mean reversion signal: {signal_side.value} @ {entry_price:.5f}, "
                f"z-score={z_score:.2f}, MA={ma:.5f}"
            )

        return signals

    def _calculate_zscore(
        self,
        data: pd.DataFrame,
    ) -> tuple[float, float, float]:
        """
        Calculate z-score of current price vs rolling mean.

        Args:
            data: Historical OHLC data

        Returns:
            Tuple of (z_score, moving_average, standard_deviation)
        """
        if self.use_returns:
            # Z-score of returns
            returns = data["close"].pct_change()
            ma = returns.rolling(self.lookback).mean().iloc[-1]
            std = returns.rolling(self.lookback).std().iloc[-1]
            current = returns.iloc[-1]
        else:
            # Z-score of price
            ma = data["close"].rolling(self.lookback).mean().iloc[-1]
            std = data["close"].rolling(self.lookback).std().iloc[-1]
            current = data["close"].iloc[-1]

        if pd.isna(std) or std == 0:
            return np.nan, ma, std

        z_score = (current - ma) / std
        return z_score, ma, std

    def get_lookback_period(self) -> int:
        """Get minimum lookback needed."""
        return self.lookback + 1

    def check_exit_signal(
        self,
        data: pd.DataFrame,
        current_index: int,
        position_side: Side,
    ) -> bool:
        """
        Check if mean reversion exit condition is met.

        Args:
            data: Historical data
            current_index: Current bar index
            position_side: Current position side

        Returns:
            True if should exit
        """
        if not self.exit_at_mean:
            return False

        historical = data.iloc[:current_index + 1]
        z_score, _, _ = self._calculate_zscore(historical)

        if pd.isna(z_score):
            return False

        # Exit when z-score crosses zero (returns to mean)
        if position_side == Side.LONG and z_score >= 0:
            return True
        elif position_side == Side.SHORT and z_score <= 0:
            return True

        return False
