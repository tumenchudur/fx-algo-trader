"""Tests for data validation."""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from fx_trading.data.validation import DataQualityValidator, DataQualityReport


class TestDataQualityValidator:
    """Tests for data quality validator."""

    @pytest.fixture
    def validator(self) -> DataQualityValidator:
        """Create validator."""
        return DataQualityValidator("EURUSD", "M5")

    def test_detects_duplicates(self, validator):
        """Should detect duplicate timestamps."""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
        timestamps = timestamps.insert(5, timestamps[4])  # Duplicate

        df = pd.DataFrame({
            "open": [1.08] * 11,
            "high": [1.081] * 11,
            "low": [1.079] * 11,
            "close": [1.0805] * 11,
        }, index=timestamps)

        report = validator.validate(df)
        assert report.duplicate_timestamps > 0

    def test_detects_negative_prices(self, validator):
        """Should detect negative prices."""
        df = pd.DataFrame({
            "open": [1.08, -1.08, 1.08],
            "high": [1.081, 1.081, 1.081],
            "low": [1.079, 1.079, 1.079],
            "close": [1.0805, 1.0805, 1.0805],
        }, index=pd.date_range("2024-01-01", periods=3, freq="5min", tz="UTC"))

        report = validator.validate(df)
        assert report.negative_prices > 0
        assert not report.is_valid

    def test_detects_invalid_ohlc(self, validator):
        """Should detect high < low."""
        df = pd.DataFrame({
            "open": [1.08, 1.08],
            "high": [1.079, 1.081],  # First has high < low
            "low": [1.081, 1.079],
            "close": [1.0805, 1.0805],
        }, index=pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"))

        report = validator.validate(df)
        assert report.invalid_ohlc > 0
        assert not report.is_valid

    def test_detects_gaps(self, validator):
        """Should detect significant gaps in timestamps."""
        # Create data with a 1 hour gap (12 missing 5-min bars)
        timestamps = pd.DatetimeIndex([
            datetime(2024, 1, 1, 0, 0),
            datetime(2024, 1, 1, 0, 5),
            datetime(2024, 1, 1, 1, 5),  # 1 hour gap
            datetime(2024, 1, 1, 1, 10),
        ], tz="UTC")

        df = pd.DataFrame({
            "open": [1.08] * 4,
            "high": [1.081] * 4,
            "low": [1.079] * 4,
            "close": [1.0805] * 4,
        }, index=timestamps)

        report = validator.validate(df)
        assert report.missing_timestamps > 0

    def test_detects_outliers(self, validator):
        """Should detect price outliers."""
        np.random.seed(42)
        n = 100
        prices = 1.08 + np.random.normal(0, 0.001, n)
        prices[50] = 1.50  # Huge outlier

        df = pd.DataFrame({
            "open": prices - 0.0001,
            "high": prices + 0.0005,
            "low": prices - 0.0005,
            "close": prices,
        }, index=pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"))

        report = validator.validate(df)
        assert report.price_outliers > 0

    def test_checks_bid_ask_validity(self, validator):
        """Should check bid < ask."""
        df = pd.DataFrame({
            "open": [1.08, 1.08],
            "high": [1.081, 1.081],
            "low": [1.079, 1.079],
            "close": [1.0805, 1.0805],
            "bid": [1.08, 1.081],  # Second: bid > ask
            "ask": [1.0802, 1.0805],
        }, index=pd.date_range("2024-01-01", periods=2, freq="5min", tz="UTC"))

        report = validator.validate(df)
        # Should have issue about invalid spread
        assert any("spread" in issue["type"].lower() for issue in report.issues)

    def test_quality_score_calculation(self, validator):
        """Should calculate quality score correctly."""
        df = pd.DataFrame({
            "open": [1.08] * 10,
            "high": [1.081] * 10,
            "low": [1.079] * 10,
            "close": [1.0805] * 10,
        }, index=pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC"))

        report = validator.validate(df)
        assert report.quality_score == 100.0
        assert report.is_valid

    def test_empty_dataframe(self, validator):
        """Should handle empty DataFrame."""
        df = pd.DataFrame()
        report = validator.validate(df)
        assert report.total_rows == 0
        assert report.quality_score == 0

    def test_clean_removes_duplicates(self, validator):
        """Cleaning should remove duplicates."""
        timestamps = pd.date_range("2024-01-01", periods=10, freq="5min", tz="UTC")
        timestamps = timestamps.insert(5, timestamps[4])

        df = pd.DataFrame({
            "open": [1.08] * 11,
            "high": [1.081] * 11,
            "low": [1.079] * 11,
            "close": [1.0805] * 11,
        }, index=timestamps)

        cleaned = validator.clean(df, remove_duplicates=True)
        assert len(cleaned) == 10
        assert not cleaned.index.duplicated().any()
