"""Tests for economic calendar and news filter."""

from datetime import datetime, timedelta
import pytest

from fx_trading.data.economic_calendar import EconomicCalendar, EconomicEvent, Impact
from fx_trading.risk.news_filter import NewsFilter, NewsFilterConfig, FilterResult, TradingSchedule


class TestEconomicEvent:
    """Tests for EconomicEvent class."""

    def test_affects_pair_matching_base(self):
        """Test event affects pair when currency is base."""
        event = EconomicEvent(
            title="Non-Farm Payrolls",
            currency="USD",
            timestamp=datetime.utcnow(),
            impact=Impact.HIGH,
        )
        assert event.affects_pair("EURUSD")
        assert event.affects_pair("EUR/USD")
        assert event.affects_pair("USDJPY")

    def test_affects_pair_matching_quote(self):
        """Test event affects pair when currency is quote."""
        event = EconomicEvent(
            title="ECB Rate Decision",
            currency="EUR",
            timestamp=datetime.utcnow(),
            impact=Impact.HIGH,
        )
        assert event.affects_pair("EURUSD")
        assert event.affects_pair("EURJPY")

    def test_affects_pair_no_match(self):
        """Test event doesn't affect unrelated pair."""
        event = EconomicEvent(
            title="BOJ Rate Decision",
            currency="JPY",
            timestamp=datetime.utcnow(),
            impact=Impact.HIGH,
        )
        assert not event.affects_pair("EURUSD")
        assert event.affects_pair("USDJPY")


class TestNewsFilter:
    """Tests for NewsFilter class."""

    def test_filter_disabled(self):
        """Test filter allows trading when disabled."""
        config = NewsFilterConfig(enabled=False)
        news_filter = NewsFilter(config)

        result = news_filter.check_trading_allowed("EURUSD")

        assert result.allowed
        assert result.reason == "Filter disabled"

    def test_filter_allows_trading_no_events(self):
        """Test filter allows trading when no blocking events."""
        config = NewsFilterConfig(
            enabled=True,
            minutes_before=30,
            minutes_after=15,
        )
        news_filter = NewsFilter(config)
        # Mock empty calendar response
        news_filter._calendar._events = []
        news_filter._calendar._last_fetch = datetime.utcnow()

        result = news_filter.check_trading_allowed("EURUSD")

        assert result.allowed

    def test_filter_blocks_before_event(self):
        """Test filter blocks trading before high-impact event."""
        config = NewsFilterConfig(
            enabled=True,
            minutes_before=30,
            minutes_after=15,
            min_impact=Impact.HIGH,
        )
        news_filter = NewsFilter(config)

        # Create event 15 minutes from now
        event_time = datetime.utcnow() + timedelta(minutes=15)
        mock_event = EconomicEvent(
            title="Non-Farm Payrolls",
            currency="USD",
            timestamp=event_time,
            impact=Impact.HIGH,
        )
        news_filter._calendar._events = [mock_event]
        news_filter._calendar._last_fetch = datetime.utcnow()

        result = news_filter.check_trading_allowed("EURUSD")

        assert not result.allowed
        assert "Non-Farm Payrolls" in result.reason
        assert result.blocking_event == mock_event

    def test_filter_blocks_after_event(self):
        """Test filter blocks trading during cooldown after event."""
        config = NewsFilterConfig(
            enabled=True,
            minutes_before=30,
            minutes_after=15,
            min_impact=Impact.HIGH,
        )
        news_filter = NewsFilter(config)

        # Create event 5 minutes ago (still in cooldown)
        event_time = datetime.utcnow() - timedelta(minutes=5)
        mock_event = EconomicEvent(
            title="FOMC Statement",
            currency="USD",
            timestamp=event_time,
            impact=Impact.HIGH,
        )
        news_filter._calendar._events = [mock_event]
        news_filter._calendar._last_fetch = datetime.utcnow()

        result = news_filter.check_trading_allowed("EURUSD")

        assert not result.allowed
        assert "cooldown" in result.reason.lower() or "FOMC" in result.reason

    def test_filter_allows_after_cooldown(self):
        """Test filter allows trading after cooldown period."""
        config = NewsFilterConfig(
            enabled=True,
            minutes_before=30,
            minutes_after=15,
            min_impact=Impact.HIGH,
        )
        news_filter = NewsFilter(config)

        # Create event 20 minutes ago (past 15 min cooldown)
        event_time = datetime.utcnow() - timedelta(minutes=20)
        mock_event = EconomicEvent(
            title="CPI Release",
            currency="USD",
            timestamp=event_time,
            impact=Impact.HIGH,
        )
        news_filter._calendar._events = [mock_event]
        news_filter._calendar._last_fetch = datetime.utcnow()

        result = news_filter.check_trading_allowed("EURUSD")

        assert result.allowed

    def test_filter_ignores_low_impact(self):
        """Test filter ignores low impact events when set to high."""
        config = NewsFilterConfig(
            enabled=True,
            minutes_before=30,
            minutes_after=15,
            min_impact=Impact.HIGH,
        )
        news_filter = NewsFilter(config)

        # Create low impact event 10 minutes from now
        event_time = datetime.utcnow() + timedelta(minutes=10)
        mock_event = EconomicEvent(
            title="Minor Data Release",
            currency="USD",
            timestamp=event_time,
            impact=Impact.LOW,
        )
        news_filter._calendar._events = [mock_event]
        news_filter._calendar._last_fetch = datetime.utcnow()

        result = news_filter.check_trading_allowed("EURUSD")

        assert result.allowed

    def test_filter_respects_currency_filter(self):
        """Test filter only blocks for specified currencies."""
        config = NewsFilterConfig(
            enabled=True,
            minutes_before=30,
            minutes_after=15,
            min_impact=Impact.HIGH,
            currencies=["EUR"],  # Only filter EUR events
        )
        news_filter = NewsFilter(config)

        # Create USD event - should be ignored due to currency filter
        event_time = datetime.utcnow() + timedelta(minutes=10)
        mock_event = EconomicEvent(
            title="Non-Farm Payrolls",
            currency="USD",
            timestamp=event_time,
            impact=Impact.HIGH,
        )
        news_filter._calendar._events = [mock_event]
        news_filter._calendar._last_fetch = datetime.utcnow()

        result = news_filter.check_trading_allowed("EURUSD")

        # Should allow because USD not in currency filter
        assert result.allowed


class TestTradingSchedule:
    """Tests for TradingSchedule class."""

    def test_market_closed_saturday(self):
        """Test market is closed on Saturday."""
        schedule = TradingSchedule(trade_weekends=False)

        # Create a Saturday timestamp
        saturday = datetime(2024, 1, 6, 12, 0)  # 2024-01-06 is a Saturday
        assert not schedule.is_market_open(saturday)

    def test_market_closed_sunday_early(self):
        """Test market is closed on Sunday before open."""
        schedule = TradingSchedule(trade_weekends=False)

        # Create early Sunday timestamp (before 10pm UTC)
        sunday_early = datetime(2024, 1, 7, 10, 0)  # 10am UTC Sunday
        assert not schedule.is_market_open(sunday_early)

    def test_market_open_weekday(self):
        """Test market is open on weekdays."""
        schedule = TradingSchedule(trade_weekends=False)

        # Create a Tuesday timestamp
        tuesday = datetime(2024, 1, 9, 12, 0)
        assert schedule.is_market_open(tuesday)

    def test_can_trade_combines_checks(self):
        """Test can_trade combines market hours and news filter."""
        config = NewsFilterConfig(enabled=False)
        news_filter = NewsFilter(config)
        schedule = TradingSchedule(news_filter=news_filter)

        # Weekday
        tuesday = datetime(2024, 1, 9, 12, 0)
        result = schedule.can_trade("EURUSD", tuesday)

        assert result.allowed

    def test_can_trade_blocks_weekend(self):
        """Test can_trade blocks on weekends."""
        schedule = TradingSchedule(trade_weekends=False)

        saturday = datetime(2024, 1, 6, 12, 0)
        result = schedule.can_trade("EURUSD", saturday)

        assert not result.allowed
        assert "weekend" in result.reason.lower()


class TestFilterResult:
    """Tests for FilterResult class."""

    def test_allowed_result(self):
        """Test allowed filter result."""
        result = FilterResult(allowed=True, reason="No blocking events")

        assert result.allowed
        assert result.blocking_event is None

    def test_blocked_result_with_event(self):
        """Test blocked filter result with event."""
        event = EconomicEvent(
            title="NFP",
            currency="USD",
            timestamp=datetime.utcnow(),
            impact=Impact.HIGH,
        )
        result = FilterResult(
            allowed=False,
            reason="Approaching NFP",
            blocking_event=event,
            resume_time=datetime.utcnow() + timedelta(minutes=45),
        )

        assert not result.allowed
        assert result.blocking_event == event
        assert result.resume_time is not None