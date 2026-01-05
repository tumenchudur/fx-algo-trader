"""
Synthetic data generator for testing and demos.

Generates realistic-looking FX OHLCV data with bid/ask.
"""

from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


class SyntheticDataGenerator:
    """
    Generate synthetic FX data for testing.

    Creates OHLCV data with bid/ask spreads that mimics real market behavior.
    """

    # Typical FX pair characteristics
    PAIR_DEFAULTS = {
        "EURUSD": {"base_price": 1.0800, "volatility": 0.0008, "spread_pips": 1.2},
        "GBPUSD": {"base_price": 1.2600, "volatility": 0.0010, "spread_pips": 1.5},
        "USDJPY": {"base_price": 150.00, "volatility": 0.0007, "spread_pips": 1.0},
        "AUDUSD": {"base_price": 0.6500, "volatility": 0.0009, "spread_pips": 1.3},
        "USDCAD": {"base_price": 1.3600, "volatility": 0.0007, "spread_pips": 1.4},
        "USDCHF": {"base_price": 0.8800, "volatility": 0.0008, "spread_pips": 1.5},
    }

    TIMEFRAME_MINUTES = {
        "M1": 1,
        "M5": 5,
        "M15": 15,
        "M30": 30,
        "H1": 60,
        "H4": 240,
        "D1": 1440,
    }

    def __init__(self, seed: Optional[int] = 42):
        """
        Initialize generator.

        Args:
            seed: Random seed for reproducibility
        """
        self.seed = seed
        if seed is not None:
            np.random.seed(seed)

    def generate(
        self,
        symbol: str = "EURUSD",
        timeframe: str = "M5",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        num_bars: int = 1000,
        base_price: Optional[float] = None,
        volatility: Optional[float] = None,
        spread_pips: Optional[float] = None,
        trend: float = 0.0,
        include_volume: bool = True,
    ) -> pd.DataFrame:
        """
        Generate synthetic OHLCV data.

        Args:
            symbol: Trading pair symbol
            timeframe: Candle timeframe (M1, M5, H1, etc.)
            start_date: Start datetime (default: 1000 bars ago)
            end_date: End datetime (default: now)
            num_bars: Number of bars to generate
            base_price: Starting price (default: pair default)
            volatility: Per-bar volatility (default: pair default)
            spread_pips: Bid-ask spread in pips (default: pair default)
            trend: Drift parameter (-1 to 1, 0 = no trend)
            include_volume: Include volume column

        Returns:
            DataFrame with OHLCV + bid/ask data
        """
        # Get defaults for pair
        defaults = self.PAIR_DEFAULTS.get(symbol, self.PAIR_DEFAULTS["EURUSD"])
        base_price = base_price or defaults["base_price"]
        volatility = volatility or defaults["volatility"]
        spread_pips = spread_pips or defaults["spread_pips"]

        # Determine pip value (most pairs are 4 decimal, JPY pairs are 2)
        pip_value = 0.01 if "JPY" in symbol else 0.0001
        spread = spread_pips * pip_value

        # Generate timestamps
        if timeframe not in self.TIMEFRAME_MINUTES:
            raise ValueError(f"Unknown timeframe: {timeframe}")

        interval = timedelta(minutes=self.TIMEFRAME_MINUTES[timeframe])

        if end_date is None:
            end_date = datetime.utcnow().replace(second=0, microsecond=0)
        if start_date is None:
            start_date = end_date - (interval * num_bars)

        timestamps = pd.date_range(start=start_date, end=end_date, freq=interval)[:num_bars]

        # Generate price series using geometric Brownian motion
        n = len(timestamps)
        drift = trend * volatility
        returns = np.random.normal(drift, volatility, n)
        returns[0] = 0  # Start at base price

        # Calculate close prices
        log_prices = np.log(base_price) + np.cumsum(returns)
        close_prices = np.exp(log_prices)

        # Generate OHLC from close
        ohlc = self._generate_ohlc(close_prices, volatility)

        # Generate bid/ask with variable spread
        bid, ask, spread_series = self._generate_bid_ask(close_prices, spread, volatility)

        # Generate volume
        volume = self._generate_volume(n, volatility, include_volume)

        # Build DataFrame
        df = pd.DataFrame({
            "open": ohlc["open"],
            "high": ohlc["high"],
            "low": ohlc["low"],
            "close": ohlc["close"],
            "bid": bid,
            "ask": ask,
            "spread": spread_series,
            "spread_pips": spread_series / pip_value,
            "mid": (bid + ask) / 2,
            "volume": volume,
            "symbol": symbol,
        }, index=pd.DatetimeIndex(timestamps, tz="UTC", name="datetime"))

        logger.info(f"Generated {len(df)} synthetic bars for {symbol} {timeframe}")
        return df

    def _generate_ohlc(
        self,
        close_prices: np.ndarray,
        volatility: float,
    ) -> dict[str, np.ndarray]:
        """Generate OHLC from close prices."""
        n = len(close_prices)

        # Generate intra-bar variation
        intra_vol = volatility * 0.5

        # Open is previous close with small gap
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        open_prices += np.random.normal(0, volatility * 0.1, n)

        # High and low based on open, close, and volatility
        high_prices = np.maximum(open_prices, close_prices)
        low_prices = np.minimum(open_prices, close_prices)

        # Extend high/low based on volatility
        high_extension = np.abs(np.random.normal(0, intra_vol, n)) * close_prices
        low_extension = np.abs(np.random.normal(0, intra_vol, n)) * close_prices

        high_prices += high_extension
        low_prices -= low_extension

        return {
            "open": open_prices,
            "high": high_prices,
            "low": low_prices,
            "close": close_prices,
        }

    def _generate_bid_ask(
        self,
        close_prices: np.ndarray,
        base_spread: float,
        volatility: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Generate bid/ask prices with variable spread."""
        n = len(close_prices)

        # Spread varies with volatility (widens in volatile periods)
        spread_variation = np.abs(np.random.normal(1, 0.2, n))
        spread_series = base_spread * spread_variation

        # Bid/ask centered on close
        ask = close_prices + spread_series / 2
        bid = close_prices - spread_series / 2

        return bid, ask, spread_series

    def _generate_volume(
        self,
        n: int,
        volatility: float,
        include: bool,
    ) -> np.ndarray:
        """Generate tick volume."""
        if not include:
            return np.zeros(n)

        # Base volume with log-normal distribution
        base_volume = np.random.lognormal(mean=8, sigma=1, size=n)

        # Volume tends to be higher during volatility
        vol_factor = 1 + np.random.uniform(0, volatility * 100, n)

        return (base_volume * vol_factor).astype(int)

    def generate_multiple_symbols(
        self,
        symbols: list[str],
        timeframe: str = "M5",
        num_bars: int = 1000,
        start_date: Optional[datetime] = None,
        correlation: float = 0.3,
    ) -> pd.DataFrame:
        """
        Generate correlated data for multiple symbols.

        Args:
            symbols: List of symbols to generate
            timeframe: Candle timeframe
            num_bars: Bars per symbol
            start_date: Start date
            correlation: Correlation between pairs (0-1)

        Returns:
            Combined DataFrame with all symbols
        """
        dfs = []

        # Generate base returns for correlation
        base_returns = np.random.normal(0, 1, num_bars)

        for i, symbol in enumerate(symbols):
            # Mix base returns with random returns for correlation
            pair_returns = correlation * base_returns + (1 - correlation) * np.random.normal(0, 1, num_bars)
            pair_returns = (pair_returns - pair_returns.mean()) / pair_returns.std()

            defaults = self.PAIR_DEFAULTS.get(symbol, self.PAIR_DEFAULTS["EURUSD"])
            volatility = defaults["volatility"]

            # Scale returns
            scaled_returns = pair_returns * volatility

            # Generate with correlated returns
            df = self.generate(
                symbol=symbol,
                timeframe=timeframe,
                num_bars=num_bars,
                start_date=start_date,
            )

            dfs.append(df)

        return pd.concat(dfs)

    def save_sample_data(
        self,
        output_dir: str = "data/sample",
        symbols: Optional[list[str]] = None,
        timeframe: str = "M5",
        num_bars: int = 5000,
    ) -> None:
        """
        Generate and save sample data files.

        Args:
            output_dir: Output directory
            symbols: Symbols to generate (default: EURUSD, GBPUSD)
            timeframe: Candle timeframe
            num_bars: Number of bars per symbol
        """
        from pathlib import Path

        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        symbols = symbols or ["EURUSD", "GBPUSD"]

        for symbol in symbols:
            df = self.generate(symbol=symbol, timeframe=timeframe, num_bars=num_bars)

            # Save as Parquet
            parquet_path = output_path / f"{symbol}_{timeframe}.parquet"
            df.to_parquet(parquet_path, index=True)
            logger.info(f"Saved {parquet_path}")

            # Save as CSV
            csv_path = output_path / f"{symbol}_{timeframe}.csv"
            df.to_csv(csv_path, index=True)
            logger.info(f"Saved {csv_path}")
