"""
Data ingestion module for OHLCV and tick data.

Supports CSV input and Parquet output.
"""

from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from fx_trading.config.models import DataConfig
from fx_trading.data.validation import DataQualityValidator, DataQualityReport


class DataIngestor:
    """
    Ingest and normalize FX data.

    Handles various CSV formats and normalizes to canonical schema.
    """

    # Standard column names
    REQUIRED_COLUMNS = ["open", "high", "low", "close"]
    OPTIONAL_COLUMNS = ["volume", "bid", "ask", "spread"]

    # Common column name variations
    COLUMN_ALIASES = {
        "open": ["open", "o", "open_price", "openprice"],
        "high": ["high", "h", "high_price", "highprice"],
        "low": ["low", "l", "low_price", "lowprice"],
        "close": ["close", "c", "close_price", "closeprice"],
        "volume": ["volume", "vol", "v", "tick_volume", "tickvol"],
        "bid": ["bid", "bid_price", "bidprice"],
        "ask": ["ask", "ask_price", "askprice", "offer"],
        "spread": ["spread", "spread_pips"],
    }

    def __init__(self, config: Optional[DataConfig] = None):
        """
        Initialize ingestor.

        Args:
            config: Optional data configuration
        """
        self.config = config

    def ingest_csv(
        self,
        path: Path,
        symbol: str,
        timeframe: str = "M5",
        column_mapping: Optional[dict[str, str]] = None,
        timezone: str = "UTC",
    ) -> tuple[pd.DataFrame, DataQualityReport]:
        """
        Ingest data from CSV file.

        Args:
            path: Path to CSV file
            symbol: Trading symbol
            timeframe: Data timeframe
            column_mapping: Optional column name mapping
            timezone: Target timezone

        Returns:
            Tuple of (DataFrame, DataQualityReport)
        """
        logger.info(f"Ingesting CSV from {path}")

        # Read CSV
        df = pd.read_csv(path)
        logger.info(f"Read {len(df)} rows, columns: {list(df.columns)}")

        # Normalize column names
        df = self._normalize_columns(df, column_mapping)

        # Parse and set datetime index
        df = self._set_datetime_index(df, timezone)

        # Add derived columns
        df = self._add_derived_columns(df, symbol)

        # Validate
        validator = DataQualityValidator(symbol, timeframe)
        report = validator.validate(df)

        # Clean if requested
        if self.config:
            df = validator.clean(
                df,
                remove_duplicates=self.config.remove_duplicates,
                remove_outliers=self.config.remove_outliers,
                fill_missing=self.config.fill_missing,
            )

        logger.info(f"Ingestion complete. Quality score: {report.quality_score:.1f}%")
        return df, report

    def _normalize_columns(
        self,
        df: pd.DataFrame,
        column_mapping: Optional[dict[str, str]] = None,
    ) -> pd.DataFrame:
        """Normalize column names to standard format."""
        df = df.copy()
        df.columns = df.columns.str.lower().str.strip()

        # Apply custom mapping first
        if column_mapping:
            df = df.rename(columns=column_mapping)

        # Apply alias mapping
        rename_map = {}
        for standard, aliases in self.COLUMN_ALIASES.items():
            for alias in aliases:
                if alias in df.columns and standard not in df.columns:
                    rename_map[alias] = standard
                    break

        df = df.rename(columns=rename_map)

        # Verify required columns
        missing = [c for c in self.REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}. Available: {list(df.columns)}")

        return df

    def _set_datetime_index(
        self,
        df: pd.DataFrame,
        timezone: str = "UTC",
    ) -> pd.DataFrame:
        """Parse datetime and set as index."""
        df = df.copy()

        # Find datetime column
        datetime_cols = ["datetime", "timestamp", "time", "date", "dt"]
        datetime_col = None

        for col in datetime_cols:
            if col in df.columns:
                datetime_col = col
                break

        if datetime_col is None:
            # Check if index is already datetime
            if isinstance(df.index, pd.DatetimeIndex):
                return df
            raise ValueError(f"No datetime column found. Available: {list(df.columns)}")

        # Parse datetime
        df[datetime_col] = pd.to_datetime(df[datetime_col], utc=True)

        # Convert timezone
        if timezone != "UTC":
            df[datetime_col] = df[datetime_col].dt.tz_convert(timezone)

        # Set index
        df = df.set_index(datetime_col)
        df.index.name = "datetime"

        # Sort
        df = df.sort_index()

        return df

    def _add_derived_columns(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Add derived columns if not present."""
        df = df.copy()

        # Add symbol
        df["symbol"] = symbol

        # Add mid price if bid/ask available
        if "bid" in df.columns and "ask" in df.columns:
            df["mid"] = (df["bid"] + df["ask"]) / 2
            df["spread"] = df["ask"] - df["bid"]
            df["spread_pips"] = df["spread"] * 10000
        else:
            # Estimate bid/ask from close
            df["mid"] = df["close"]
            # Bid/ask will be derived from spread model during execution

        return df

    def save_parquet(
        self,
        df: pd.DataFrame,
        path: Path,
        compression: str = "snappy",
    ) -> None:
        """
        Save DataFrame to Parquet format.

        Args:
            df: DataFrame to save
            path: Output path
            compression: Compression algorithm
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        df.to_parquet(path, compression=compression, index=True)
        logger.info(f"Saved {len(df)} rows to {path}")

    def load_parquet(self, path: Path) -> pd.DataFrame:
        """
        Load DataFrame from Parquet file.

        Args:
            path: Path to Parquet file

        Returns:
            DataFrame
        """
        df = pd.read_parquet(path)
        logger.info(f"Loaded {len(df)} rows from {path}")
        return df

    def load_data(
        self,
        path: Path,
        symbol: Optional[str] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
    ) -> pd.DataFrame:
        """
        Load data from file (auto-detect format).

        Args:
            path: Path to data file
            symbol: Optional symbol filter
            start_date: Optional start date filter
            end_date: Optional end date filter

        Returns:
            Filtered DataFrame
        """
        path = Path(path)

        if path.suffix == ".parquet":
            df = self.load_parquet(path)
        elif path.suffix == ".csv":
            df, _ = self.ingest_csv(path, symbol or "UNKNOWN")
        else:
            raise ValueError(f"Unsupported file format: {path.suffix}")

        # Filter by symbol
        if symbol and "symbol" in df.columns:
            df = df[df["symbol"] == symbol]

        # Filter by date
        if start_date:
            if start_date.tzinfo is None:
                start_date = start_date.replace(tzinfo=df.index.tzinfo)
            df = df[df.index >= start_date]

        if end_date:
            if end_date.tzinfo is None:
                end_date = end_date.replace(tzinfo=df.index.tzinfo)
            df = df[df.index <= end_date]

        return df
