"""News filter for blocking trades around economic events.

Prevents trading during high-impact news releases to avoid
unpredictable volatility and slippage.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Optional

from fx_trading.data.economic_calendar import (
    EconomicCalendar,
    EconomicEvent,
    Impact,
)

logger = logging.getLogger(__name__)


@dataclass
class NewsFilterConfig:
    """Configuration for news filter behavior."""

    # Enable/disable the filter
    enabled: bool = True

    # Minutes before event to stop trading
    minutes_before: int = 30

    # Minutes after event to resume trading
    minutes_after: int = 15

    # Minimum impact level to filter (HIGH = only major events)
    min_impact: Impact = Impact.HIGH

    # Currencies to monitor (None = all currencies in traded pairs)
    currencies: Optional[list[str]] = None

    # Whether to also block position modifications (stop loss changes)
    block_modifications: bool = False


@dataclass
class FilterResult:
    """Result of news filter check."""

    allowed: bool
    reason: str
    blocking_event: Optional[EconomicEvent] = None
    resume_time: Optional[datetime] = None


class NewsFilter:
    """Filters trading around high-impact economic events.

    Usage:
        filter = NewsFilter(config)
        result = filter.check_trading_allowed("EUR/USD")
        if not result.allowed:
            logger.info(f"Trading blocked: {result.reason}")
    """

    def __init__(self, config: Optional[NewsFilterConfig] = None):
        self.config = config or NewsFilterConfig()
        self._blocked_until: dict[str, datetime] = {}

    def check_trading_allowed(
        self,
        pair: str,
        now: Optional[datetime] = None,
    ) -> FilterResult:
        """Check if trading is allowed for a currency pair.

        Args:
            pair: Currency pair (e.g., "EUR/USD")
            now: Current time (defaults to UTC now)

        Returns:
            FilterResult with allowed status and details
        """
        if not self.config.enabled:
            return FilterResult(allowed=True, reason="Filter disabled")

        now = now or datetime.utcnow()

        # Check if we're still in a blocked period from previous event
        if pair in self._blocked_until:
            if now < self._blocked_until[pair]:
                return FilterResult(
                    allowed=False,
                    reason="Still in post-event cooldown",
                    resume_time=self._blocked_until[pair],
                )
            else:
                del self._blocked_until[pair]

        # Get currencies from pair
        pair_currencies = self._extract_currencies(pair)

        # Filter currencies if config specifies
        if self.config.currencies:
            pair_currencies = [
                c for c in pair_currencies
                if c in self.config.currencies
            ]

        if not pair_currencies:
            return FilterResult(allowed=True, reason="No monitored currencies in pair")

        # Look ahead for events
        look_ahead_hours = (self.config.minutes_before + 60) / 60
        events = self._calendar.get_upcoming_events(
            hours_ahead=look_ahead_hours,
            min_impact=self.config.min_impact,
            currencies=pair_currencies,
        )

        # Check each event
        for event in events:
            time_to_event = (event.timestamp - now).total_seconds() / 60

            # Before event window
            if 0 <= time_to_event <= self.config.minutes_before:
                return FilterResult(
                    allowed=False,
                    reason=f"Approaching {event.title} ({event.currency}) in {int(time_to_event)} min",
                    blocking_event=event,
                    resume_time=event.timestamp + timedelta(minutes=self.config.minutes_after),
                )

            # During/after event window (event just happened)
            if -self.config.minutes_after <= time_to_event < 0:
                resume_time = event.timestamp + timedelta(minutes=self.config.minutes_after)
                self._blocked_until[pair] = resume_time
                return FilterResult(
                    allowed=False,
                    reason=f"Post-event cooldown: {event.title} ({event.currency})",
                    blocking_event=event,
                    resume_time=resume_time,
                )

        return FilterResult(allowed=True, reason="No blocking events")

    def get_next_event(
        self,
        pair: Optional[str] = None,
        hours_ahead: float = 24,
    ) -> Optional[EconomicEvent]:
        """Get the next high-impact event.

        Args:
            pair: Optional pair to filter by
            hours_ahead: How far ahead to look

        Returns:
            Next event or None
        """
        currencies = None
        if pair:
            currencies = self._extract_currencies(pair)
            if self.config.currencies:
                currencies = [c for c in currencies if c in self.config.currencies]

        events = self._calendar.get_upcoming_events(
            hours_ahead=hours_ahead,
            min_impact=self.config.min_impact,
            currencies=currencies,
        )

        return events[0] if events else None

    def get_todays_events(
        self,
        min_impact: Optional[Impact] = None,
    ) -> list[EconomicEvent]:
        """Get all events for today.

        Args:
            min_impact: Minimum impact level (defaults to config)

        Returns:
            List of today's events
        """
        return self._calendar.get_upcoming_events(
            hours_ahead=24,
            min_impact=min_impact or self.config.min_impact,
        )

    def _extract_currencies(self, pair: str) -> list[str]:
        """Extract currency codes from pair string."""
        # Handle various formats: EUR/USD, EURUSD, EUR_USD
        clean = pair.upper().replace("/", "").replace("_", "")
        if len(clean) == 6:
            return [clean[:3], clean[3:]]
        return []

    def refresh_calendar(self):
        """Force refresh of economic calendar data."""
        self._calendar.fetch_events(force=True)

    def close(self):
        """Clean up resources."""
        self._calendar.close()


class TradingSchedule:
    """Manages trading hours and combines with news filter.

    Provides unified check for whether trading is allowed
    based on market hours, news events, and custom rules.
    """

    # Forex market hours (Sunday 5pm ET to Friday 5pm ET)
    # Represented in UTC
    MARKET_OPEN_HOUR_UTC = 22  # Sunday 10pm UTC (5pm ET)
    MARKET_CLOSE_DAY = 4  # Friday
    MARKET_CLOSE_HOUR_UTC = 22  # Friday 10pm UTC

    def __init__(
        self,
        news_filter: Optional[NewsFilter] = None,
        trade_weekends: bool = False,
    ):
        self.news_filter = news_filter
        self.trade_weekends = trade_weekends

    def is_market_open(self, now: Optional[datetime] = None) -> bool:
        """Check if forex market is open."""
        now = now or datetime.utcnow()

        # Weekend check (Saturday or Sunday before open)
        if now.weekday() == 5:  # Saturday
            return False
        if now.weekday() == 6 and now.hour < self.MARKET_OPEN_HOUR_UTC:  # Sunday before open
            return False
        if now.weekday() == 4 and now.hour >= self.MARKET_CLOSE_HOUR_UTC:  # Friday after close
            return False

        return True

    def can_trade(self, pair: str, now: Optional[datetime] = None) -> FilterResult:
        """Unified check for trading permission.

        Combines market hours and news filter.
        """
        now = now or datetime.utcnow()

        # Check market hours
        if not self.trade_weekends and not self.is_market_open(now):
            return FilterResult(
                allowed=False,
                reason="Market closed (weekend)",
            )

        # Check news filter
        if self.news_filter:
            return self.news_filter.check_trading_allowed(pair, now)

        return FilterResult(allowed=True, reason="Trading allowed")