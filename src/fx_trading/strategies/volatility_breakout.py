"""
Volatility Breakout Strategy with Trend Filter.

Trades breakouts of recent highs/lows when volatility expands,
but ONLY in the direction of the higher timeframe trend.
"""

from datetime import datetime, time
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger

from fx_trading.config.models import StrategyConfig
from fx_trading.strategies.base import Strategy
from fx_trading.types.models import Signal, Side


class VolatilityBreakoutStrategy(Strategy):
    """
    Volatility Breakout Strategy with Trend Filter.

    Entry Logic:
    - Long: Price breaks above the high of the lookback period
            AND ATR is above average (volatility expanding)
            AND price is above EMA (uptrend)
            AND (optional) ADX > threshold (trending market)
    - Short: Price breaks below the low of the lookback period
             AND ATR is above average
             AND price is below EMA (downtrend)
             AND (optional) ADX > threshold

    Exit Logic:
    - Stop loss at entry minus/plus N*ATR
    - Take profit at risk-reward ratio
    - Optional time-based exit

    Parameters (via config.params):
    - lookback: Period for high/low calculation (default: 20)
    - atr_period: ATR calculation period (default: 14)
    - atr_threshold: ATR multiple above average to trade (default: 1.0)
    - sl_atr_multiplier: ATR multiplier for stop loss (default: 2.0)
    - trend_ema_period: EMA period for trend filter (default: 50)
    - use_trend_filter: Enable/disable trend filter (default: True)
    - adx_period: ADX calculation period (default: 14)
    - adx_threshold: Minimum ADX to trade (default: 20)
    - use_adx_filter: Enable/disable ADX filter (default: True)
    - use_time_filter: Enable/disable trading hours filter (default: True)
    - trading_start_hour: UTC hour to start trading (default: 7)
    - trading_end_hour: UTC hour to stop trading (default: 20)
    """

    def __init__(self, config: StrategyConfig):
        """Initialize with config."""
        super().__init__(config)

        # Core parameters
        self.lookback = self.params.get("lookback", 20)
        self.atr_period = self.params.get("atr_period", 14)
        self.atr_threshold = self.params.get("atr_threshold", 1.0)
        self.sl_atr_multiplier = self.params.get("sl_atr_multiplier", 2.0)
        self.require_close_above = self.params.get("require_close_above", True)

        # Trend filter parameters
        self.trend_ema_period = self.params.get("trend_ema_period", 50)
        self.use_trend_filter = self.params.get("use_trend_filter", True)

        # ADX filter parameters
        self.adx_period = self.params.get("adx_period", 14)
        self.adx_threshold = self.params.get("adx_threshold", 20)
        self.use_adx_filter = self.params.get("use_adx_filter", True)

        # Time filter parameters
        self.use_time_filter = self.params.get("use_time_filter", True)
        self.trading_start_hour = self.params.get("trading_start_hour", 7)  # 7 AM UTC
        self.trading_end_hour = self.params.get("trading_end_hour", 20)  # 8 PM UTC

        # RSI filter parameters
        self.use_rsi_filter = self.params.get("use_rsi_filter", True)
        self.rsi_period = self.params.get("rsi_period", 14)
        self.rsi_overbought = self.params.get("rsi_overbought", 70)
        self.rsi_oversold = self.params.get("rsi_oversold", 30)

        logger.info(
            f"VolatilityBreakout params: lookback={self.lookback}, "
            f"atr_period={self.atr_period}, atr_threshold={self.atr_threshold}, "
            f"trend_ema={self.trend_ema_period}, use_trend_filter={self.use_trend_filter}, "
            f"adx_threshold={self.adx_threshold}, use_adx_filter={self.use_adx_filter}, "
            f"time_filter={self.use_time_filter} ({self.trading_start_hour}-{self.trading_end_hour} UTC), "
            f"rsi_filter={self.use_rsi_filter} (period={self.rsi_period}, OB={self.rsi_overbought}, OS={self.rsi_oversold})"
        )

    def generate_signals(
        self,
        data: pd.DataFrame,
        current_index: int,
    ) -> list[Signal]:
        """
        Generate breakout signals with trend filter.

        Uses only data up to current_index (no lookahead).
        """
        signals = []

        if not self.enabled:
            return signals

        if not self.validate_data(data):
            return signals

        # Need enough data for indicators
        min_required = max(self.lookback, self.atr_period * 2, self.trend_ema_period) + 1
        if current_index < min_required:
            return signals

        # Get current bar and historical data (NO LOOKAHEAD)
        current_bar = data.iloc[current_index]
        historical = data.iloc[:current_index + 1]

        # Extract symbol for per-symbol parameter lookups
        symbol = current_bar.get("symbol", self.symbols[0]) if hasattr(current_bar, "get") else self.symbols[0]
        if hasattr(symbol, "iloc"):
            symbol = symbol.iloc[0] if len(symbol) > 0 else self.symbols[0]

        # TIME FILTER: Check if we're in trading hours (supports per-symbol override)
        use_time_filter = self.get_param("use_time_filter", symbol, self.use_time_filter)
        if use_time_filter:
            current_time = data.index[current_index]
            if isinstance(current_time, datetime):
                hour = current_time.hour
                if not (self.trading_start_hour <= hour < self.trading_end_hour):
                    return signals  # Outside trading hours

        # Calculate ATR indicators using only past data
        atr, atr_series = self._calculate_atr(historical)
        avg_atr = atr_series.rolling(self.atr_period * 2).mean().iloc[-1]

        if pd.isna(atr) or pd.isna(avg_atr):
            return signals

        # Check volatility condition
        volatility_expanding = atr >= avg_atr * self.atr_threshold

        if not volatility_expanding:
            return signals

        # TREND FILTER: Calculate EMA and check trend direction
        trend_direction = None
        if self.use_trend_filter:
            ema = historical["close"].ewm(span=self.trend_ema_period, adjust=False).mean()
            current_ema = ema.iloc[-1]
            current_close = current_bar["close"]

            if current_close > current_ema:
                trend_direction = "up"
            elif current_close < current_ema:
                trend_direction = "down"
            else:
                return signals  # No clear trend

        # ADX FILTER: Check if market is trending (supports per-symbol override)
        use_adx_filter = self.get_param("use_adx_filter", symbol, self.use_adx_filter)
        adx = None  # Initialize for case when filter is disabled
        if use_adx_filter:
            adx = self._calculate_adx(historical)
            if pd.isna(adx) or adx < self.adx_threshold:
                return signals  # Market not trending enough

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

        # Long breakout - only if trend is UP (or trend filter disabled)
        if current_high > upper_level:
            if not self.use_trend_filter or trend_direction == "up":
                if self.require_close_above:
                    if current_close > upper_level:
                        signal_side = Side.LONG
                else:
                    signal_side = Side.LONG

        # Short breakout - only if trend is DOWN (or trend filter disabled)
        elif current_low < lower_level:
            if not self.use_trend_filter or trend_direction == "down":
                if self.require_close_above:
                    if current_close < lower_level:
                        signal_side = Side.SHORT
                else:
                    signal_side = Side.SHORT

        # RSI FILTER: Block overbought longs and oversold shorts
        rsi = None
        if signal_side is not None and self.use_rsi_filter:
            rsi = self._calculate_rsi(historical)
            if not pd.isna(rsi):
                if signal_side == Side.LONG and rsi > self.rsi_overbought:
                    logger.debug(f"RSI filter blocked LONG: RSI={rsi:.1f} > {self.rsi_overbought}")
                    return signals  # Block overbought longs
                if signal_side == Side.SHORT and rsi < self.rsi_oversold:
                    logger.debug(f"RSI filter blocked SHORT: RSI={rsi:.1f} < {self.rsi_oversold}")
                    return signals  # Block oversold shorts

        if signal_side is not None:
            # Calculate entry price (use close for signal, actual entry uses bid/ask)
            entry_price = current_close

            # Calculate stop loss - use per-symbol sl_atr_multiplier if available
            sl_atr_mult = self.get_param("sl_atr_multiplier", symbol=symbol, default=self.sl_atr_multiplier)
            sl_distance = atr * sl_atr_mult
            if signal_side == Side.LONG:
                stop_loss = entry_price - sl_distance
            else:
                stop_loss = entry_price + sl_distance

            # Calculate take profit
            take_profit = self.calculate_take_profit(entry_price, stop_loss, signal_side)

            # Calculate signal strength based on multiple factors
            strength = self._calculate_signal_strength(atr, avg_atr, adx if use_adx_filter else 25)

            signal = Signal(
                timestamp=data.index[current_index] if isinstance(data.index[current_index], datetime) else datetime.utcnow(),
                symbol=symbol,
                side=signal_side,
                strength=strength,
                stop_loss=stop_loss,
                take_profit=take_profit,
                time_exit_bars=self.config.time_exit_bars,
                metadata={
                    "upper_level": upper_level,
                    "lower_level": lower_level,
                    "atr": atr,
                    "avg_atr": avg_atr,
                    "trend_direction": trend_direction,
                    "adx": adx if use_adx_filter else None,
                    "ema": current_ema if self.use_trend_filter else None,
                    "rsi": rsi if self.use_rsi_filter else None,
                },
            )
            signals.append(signal)

            rsi_str = f"{rsi:.1f}" if rsi is not None else "N/A"
            logger.debug(
                f"Breakout signal: {signal_side.value} @ {entry_price:.5f}, "
                f"SL={stop_loss:.5f}, TP={take_profit}, "
                f"trend={trend_direction}, ADX={adx if self.use_adx_filter else 'N/A'}, "
                f"RSI={rsi_str}"
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

    def _calculate_adx(self, data: pd.DataFrame) -> float:
        """
        Calculate ADX (Average Directional Index).

        ADX measures trend strength (not direction).
        - ADX < 20: Weak trend or ranging
        - ADX 20-40: Trending
        - ADX > 40: Strong trend

        Args:
            data: Historical OHLC data

        Returns:
            Current ADX value
        """
        high = data["high"]
        low = data["low"]
        close = data["close"]

        # Calculate +DM and -DM
        plus_dm = high.diff()
        minus_dm = -low.diff()

        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0

        # When +DM > -DM, -DM = 0 and vice versa
        plus_dm[(plus_dm < minus_dm) | (plus_dm < 0)] = 0
        minus_dm[(minus_dm < plus_dm) | (minus_dm < 0)] = 0

        # Calculate True Range
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

        # Smoothed values
        atr = tr.rolling(self.adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(self.adx_period).mean() / atr)

        # Calculate DX and ADX
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
        adx = dx.rolling(self.adx_period).mean()

        return adx.iloc[-1]

    def _calculate_signal_strength(self, atr: float, avg_atr: float, adx: float) -> float:
        """
        Calculate signal strength based on multiple factors.

        Args:
            atr: Current ATR
            avg_atr: Average ATR
            adx: Current ADX

        Returns:
            Signal strength between 0 and 1
        """
        # Volatility component (0.4 weight)
        vol_score = min(1.0, (atr / avg_atr - 1) / 0.5) * 0.4

        # Trend strength component (0.6 weight)
        adx_score = min(1.0, (adx - 20) / 30) * 0.6

        return max(0.1, min(1.0, vol_score + adx_score))

    def _calculate_rsi(self, data: pd.DataFrame) -> float:
        """
        Calculate RSI (Relative Strength Index).

        RSI measures momentum and identifies overbought/oversold conditions.
        - RSI > 70: Overbought (avoid longs)
        - RSI < 30: Oversold (avoid shorts)

        Args:
            data: Historical OHLC data

        Returns:
            Current RSI value (0-100)
        """
        close = data["close"]
        delta = close.diff()

        gain = delta.where(delta > 0, 0.0)
        loss = (-delta.where(delta < 0, 0.0))

        avg_gain = gain.rolling(window=self.rsi_period).mean()
        avg_loss = loss.rolling(window=self.rsi_period).mean()

        # Avoid division by zero
        rs = avg_gain / avg_loss.replace(0, np.inf)
        rsi = 100 - (100 / (1 + rs))

        return rsi.iloc[-1]

    def get_lookback_period(self) -> int:
        """Get minimum lookback needed."""
        return max(self.lookback, self.atr_period * 2, self.trend_ema_period, self.adx_period * 2, self.rsi_period) + 1
