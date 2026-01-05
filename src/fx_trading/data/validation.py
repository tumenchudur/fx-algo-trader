"""
Data quality validation for OHLCV data.

Ensures data integrity before backtesting.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

import numpy as np
import pandas as pd
from loguru import logger


@dataclass
class DataQualityReport:
    """Report of data quality issues found."""

    symbol: str
    timeframe: str
    total_rows: int
    valid_rows: int
    issues: list[dict] = field(default_factory=list)

    # Specific counts
    missing_timestamps: int = 0
    duplicate_timestamps: int = 0
    price_outliers: int = 0
    negative_prices: int = 0
    invalid_ohlc: int = 0  # high < low, etc.
    zero_volume: int = 0
    stale_periods: int = 0
    missing_bid_ask: int = 0

    @property
    def is_valid(self) -> bool:
        """Check if data passes quality checks."""
        critical_issues = (
            self.duplicate_timestamps
            + self.negative_prices
            + self.invalid_ohlc
        )
        return critical_issues == 0

    @property
    def quality_score(self) -> float:
        """Calculate data quality score (0-100)."""
        if self.total_rows == 0:
            return 0.0
        return (self.valid_rows / self.total_rows) * 100

    def to_dict(self) -> dict:
        """Convert to dictionary."""
        return {
            "symbol": self.symbol,
            "timeframe": self.timeframe,
            "total_rows": self.total_rows,
            "valid_rows": self.valid_rows,
            "quality_score": round(self.quality_score, 2),
            "is_valid": self.is_valid,
            "missing_timestamps": self.missing_timestamps,
            "duplicate_timestamps": self.duplicate_timestamps,
            "price_outliers": self.price_outliers,
            "negative_prices": self.negative_prices,
            "invalid_ohlc": self.invalid_ohlc,
            "zero_volume": self.zero_volume,
            "stale_periods": self.stale_periods,
            "missing_bid_ask": self.missing_bid_ask,
            "issues": self.issues,
        }


class DataQualityValidator:
    """
    Validates OHLCV data quality.

    Checks for common data issues that can corrupt backtests.
    """

    # Expected timeframe intervals
    TIMEFRAME_INTERVALS = {
        "M1": timedelta(minutes=1),
        "M5": timedelta(minutes=5),
        "M15": timedelta(minutes=15),
        "M30": timedelta(minutes=30),
        "H1": timedelta(hours=1),
        "H4": timedelta(hours=4),
        "D1": timedelta(days=1),
    }

    def __init__(
        self,
        symbol: str,
        timeframe: str,
        outlier_std_threshold: float = 5.0,
        max_gap_multiple: int = 10,
    ):
        """
        Initialize validator.

        Args:
            symbol: Trading symbol
            timeframe: Data timeframe (M1, M5, H1, etc.)
            outlier_std_threshold: Std deviations for outlier detection
            max_gap_multiple: Max gap as multiple of expected interval
        """
        self.symbol = symbol
        self.timeframe = timeframe
        self.outlier_std_threshold = outlier_std_threshold
        self.max_gap_multiple = max_gap_multiple

        if timeframe not in self.TIMEFRAME_INTERVALS:
            logger.warning(f"Unknown timeframe {timeframe}, using M5 interval for gap detection")
        self.expected_interval = self.TIMEFRAME_INTERVALS.get(timeframe, timedelta(minutes=5))

    def validate(self, df: pd.DataFrame) -> DataQualityReport:
        """
        Run all validation checks on dataframe.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataQualityReport with findings
        """
        report = DataQualityReport(
            symbol=self.symbol,
            timeframe=self.timeframe,
            total_rows=len(df),
            valid_rows=len(df),
        )

        if df.empty:
            report.issues.append({"type": "empty_data", "message": "DataFrame is empty"})
            report.valid_rows = 0
            return report

        # Run checks
        self._check_index(df, report)
        self._check_duplicates(df, report)
        self._check_missing_timestamps(df, report)
        self._check_price_validity(df, report)
        self._check_ohlc_consistency(df, report)
        self._check_outliers(df, report)
        self._check_bid_ask(df, report)
        self._check_volume(df, report)

        return report

    def _check_index(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check that index is datetime and sorted."""
        if not isinstance(df.index, pd.DatetimeIndex):
            report.issues.append({
                "type": "index_type",
                "message": "Index is not DatetimeIndex",
            })

        if not df.index.is_monotonic_increasing:
            report.issues.append({
                "type": "unsorted",
                "message": "Index is not sorted in ascending order",
            })

    def _check_duplicates(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for duplicate timestamps."""
        duplicates = df.index.duplicated()
        num_duplicates = duplicates.sum()

        if num_duplicates > 0:
            report.duplicate_timestamps = int(num_duplicates)
            report.valid_rows -= num_duplicates
            report.issues.append({
                "type": "duplicates",
                "message": f"Found {num_duplicates} duplicate timestamps",
                "first_duplicate": str(df.index[duplicates][0]) if num_duplicates > 0 else None,
            })

    def _check_missing_timestamps(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for gaps in timestamps."""
        if len(df) < 2:
            return

        time_diffs = df.index.to_series().diff()
        max_allowed_gap = self.expected_interval * self.max_gap_multiple

        gaps = time_diffs[time_diffs > max_allowed_gap]
        num_gaps = len(gaps)

        if num_gaps > 0:
            report.missing_timestamps = num_gaps
            report.stale_periods = num_gaps
            report.issues.append({
                "type": "gaps",
                "message": f"Found {num_gaps} significant gaps (>{max_allowed_gap})",
                "largest_gap": str(gaps.max()),
            })

    def _check_price_validity(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for invalid price values."""
        price_cols = ["open", "high", "low", "close"]
        existing_cols = [c for c in price_cols if c in df.columns]

        for col in existing_cols:
            negative = (df[col] <= 0).sum()
            if negative > 0:
                report.negative_prices += int(negative)
                report.valid_rows -= negative
                report.issues.append({
                    "type": "negative_price",
                    "message": f"{col} has {negative} non-positive values",
                })

            null_count = df[col].isnull().sum()
            if null_count > 0:
                report.issues.append({
                    "type": "null_price",
                    "message": f"{col} has {null_count} null values",
                })

    def _check_ohlc_consistency(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check OHLC logical consistency (high >= low, etc.)."""
        required = ["open", "high", "low", "close"]
        if not all(c in df.columns for c in required):
            return

        # High should be >= low
        invalid_hl = (df["high"] < df["low"]).sum()
        if invalid_hl > 0:
            report.invalid_ohlc += int(invalid_hl)
            report.valid_rows -= invalid_hl
            report.issues.append({
                "type": "invalid_ohlc",
                "message": f"{invalid_hl} rows where high < low",
            })

        # High should be >= open and close
        invalid_high = ((df["high"] < df["open"]) | (df["high"] < df["close"])).sum()
        if invalid_high > 0:
            report.invalid_ohlc += int(invalid_high)
            report.issues.append({
                "type": "invalid_ohlc",
                "message": f"{invalid_high} rows where high < open or close",
            })

        # Low should be <= open and close
        invalid_low = ((df["low"] > df["open"]) | (df["low"] > df["close"])).sum()
        if invalid_low > 0:
            report.invalid_ohlc += int(invalid_low)
            report.issues.append({
                "type": "invalid_ohlc",
                "message": f"{invalid_low} rows where low > open or close",
            })

    def _check_outliers(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check for price outliers using z-score."""
        if "close" not in df.columns:
            return

        returns = df["close"].pct_change().dropna()
        if len(returns) < 10:
            return

        mean_ret = returns.mean()
        std_ret = returns.std()

        if std_ret == 0:
            return

        z_scores = np.abs((returns - mean_ret) / std_ret)
        outliers = (z_scores > self.outlier_std_threshold).sum()

        if outliers > 0:
            report.price_outliers = int(outliers)
            report.issues.append({
                "type": "outliers",
                "message": f"Found {outliers} return outliers (>{self.outlier_std_threshold} std)",
            })

    def _check_bid_ask(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check bid/ask data availability and validity."""
        has_bid = "bid" in df.columns
        has_ask = "ask" in df.columns

        if not has_bid or not has_ask:
            report.missing_bid_ask = len(df)
            report.issues.append({
                "type": "missing_bid_ask",
                "message": "Bid/ask columns not present, will use spread model",
            })
            return

        # Check bid < ask
        invalid_spread = (df["bid"] >= df["ask"]).sum()
        if invalid_spread > 0:
            report.issues.append({
                "type": "invalid_spread",
                "message": f"{invalid_spread} rows where bid >= ask",
            })

        # Check for null bid/ask
        null_bid_ask = df["bid"].isnull().sum() + df["ask"].isnull().sum()
        if null_bid_ask > 0:
            report.missing_bid_ask = int(null_bid_ask // 2)
            report.issues.append({
                "type": "null_bid_ask",
                "message": f"{null_bid_ask} null bid/ask values",
            })

    def _check_volume(self, df: pd.DataFrame, report: DataQualityReport) -> None:
        """Check volume data."""
        if "volume" not in df.columns:
            return

        zero_vol = (df["volume"] == 0).sum()
        if zero_vol > 0:
            report.zero_volume = int(zero_vol)
            # Zero volume is often normal for FX
            report.issues.append({
                "type": "zero_volume",
                "message": f"{zero_vol} rows with zero volume (may be normal for FX)",
            })

    def clean(
        self,
        df: pd.DataFrame,
        remove_duplicates: bool = True,
        remove_outliers: bool = False,
        fill_missing: bool = False,
    ) -> pd.DataFrame:
        """
        Clean dataframe based on validation findings.

        Args:
            df: DataFrame to clean
            remove_duplicates: Remove duplicate timestamps
            remove_outliers: Remove price outliers
            fill_missing: Forward-fill missing values

        Returns:
            Cleaned DataFrame
        """
        df = df.copy()

        # Sort by index
        if not df.index.is_monotonic_increasing:
            df = df.sort_index()

        # Remove duplicates
        if remove_duplicates:
            df = df[~df.index.duplicated(keep="first")]

        # Remove outliers
        if remove_outliers and "close" in df.columns:
            returns = df["close"].pct_change()
            mean_ret = returns.mean()
            std_ret = returns.std()

            if std_ret > 0:
                z_scores = np.abs((returns - mean_ret) / std_ret)
                df = df[z_scores <= self.outlier_std_threshold]

        # Fill missing
        if fill_missing:
            df = df.ffill()

        logger.info(f"Cleaned data: {len(df)} rows remaining")
        return df
