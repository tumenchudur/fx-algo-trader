"""Economic calendar data fetcher.

Fetches high-impact economic events to filter trading around news releases.
"""

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum
from typing import Optional
import xml.etree.ElementTree as ET

import httpx

logger = logging.getLogger(__name__)


class Impact(Enum):
    """Economic event impact level."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


@dataclass
class EconomicEvent:
    """Single economic calendar event."""
    title: str
    currency: str
    timestamp: datetime
    impact: Impact
    forecast: Optional[str] = None
    previous: Optional[str] = None

    def affects_pair(self, pair: str) -> bool:
        """Check if this event affects a currency pair."""
        pair_upper = pair.upper().replace("/", "").replace("_", "")
        return self.currency.upper() in pair_upper


class EconomicCalendar:
    """Fetches and caches economic calendar events."""

    FOREX_FACTORY_URL = "https://www.forexfactory.com/ffcal_week_this.xml"
    CACHE_DURATION = timedelta(hours=4)

    def __init__(self):
        """Initialize economic calendar."""
        self._events: list[EconomicEvent] = []
        self._last_fetch: Optional[datetime] = None
        self._http_client = httpx.Client(
            timeout=10.0,
        )

    def fetch_events(self, force: bool = False) -> list[EconomicEvent]:
        """Fetch economic events from calendar source.

        Args:
            force: Force refresh even if cache is valid

        Returns:
            List of economic events
        """
        now = datetime.utcnow()

        # Return cached if still valid
        if not force and self._last_fetch:
            if now - self._last_fetch < self.CACHE_DURATION:
                return self._events

        try:
            events = self._fetch_forex_factory()
            self._events = events
            self._last_fetch = now
        except Exception as e:
            logger.warning(f"Failed to fetch economic calendar: {e}")
            # Return cached events on failure
        return self._events

    def _fetch_forex_factory(self) -> list[EconomicEvent]:
        """Fetch and parse Forex Factory XML feed."""
        response = self._http_client.get(self.FOREX_FACTORY_URL)
        response.raise_for_status()

        events = []
        root = ET.fromstring(response.text)

        for event_elem in root.findall(".//event"):
            try:
                event = self._parse_event(event_elem)
                if event:
                    events.append(event)
            except Exception as e:
                logger.debug(f"Failed to parse event: {e}")
                continue

        return events

    def _parse_event(self, elem) -> Optional[EconomicEvent]:
        """Parse a single event element."""
        title = elem.findtext("title", "")
        currency = elem.findtext("country", "")
        date_str = elem.findtext("date", "")
        time_str = elem.findtext("time", "")
        impact_str = elem.findtext("impact", "low")

        if not all([title, currency, date_str]):
            return None

        # Skip "All Day" or "Tentative" events
        if time_str in ["All Day", "Tentative", ""]:
            time_str = "00:00"

        # Parse timestamp (Forex Factory uses ET timezone)
        try:
            datetime_str = f"{date_str} {time_str}"
            if "am" in time_str.lower() or "pm" in time_str.lower():
                timestamp = datetime.strptime(datetime_str, "%m-%d-%Y %I:%M%p")
            else:
                timestamp = datetime.strptime(datetime_str, "%m-%d-%Y %H:%M")
        except ValueError:
            return None

        impact_map = {
            "high": Impact.HIGH,
            "medium": Impact.MEDIUM,
            "low": Impact.LOW,
        }
        impact = impact_map.get(impact_str.lower(), Impact.LOW)

        return EconomicEvent(
            title=title,
            currency=currency,
            timestamp=timestamp,
            impact=impact,
            forecast=elem.findtext("forecast"),
            previous=elem.findtext("previous"),
        )

    def get_upcoming_events(
        self,
        hours_ahead: float = 24,
        min_impact: Impact = Impact.MEDIUM,
        currencies: Optional[list[str]] = None,
        include_recent_minutes: int = 30,
    ) -> list[EconomicEvent]:
        """Get upcoming events within time window.

        Args:
            hours_ahead: Look ahead window in hours
            min_impact: Minimum impact level to include
            currencies: Filter by these currencies (e.g., ["USD", "EUR"])
            include_recent_minutes: Also include events from last N minutes

        Returns:
            Filtered list of upcoming events
        """
        events = self.fetch_events()
        now = datetime.utcnow()
        cutoff = now + timedelta(hours=hours_ahead)
        lookback = now - timedelta(minutes=include_recent_minutes)

        impact_order = {Impact.LOW: 0, Impact.MEDIUM: 1, Impact.HIGH: 2}
        min_impact_value = impact_order[min_impact]

        filtered = []
        for event in events:
            if event.timestamp < lookback or event.timestamp > cutoff:
                continue
            if impact_order[event.impact] < min_impact_value:
                continue
            if currencies:
                if event.currency.upper() not in [c.upper() for c in currencies]:
                    continue
            filtered.append(event)

        return sorted(filtered, key=lambda e: e.timestamp)

    def close(self):
        """Close HTTP client."""
        self._http_client.close()


KNOWN_HIGH_IMPACT_EVENTS = [
    # US Events
    ("Non-Farm Payrolls", "USD"),
    ("FOMC Statement", "USD"),
    ("Fed Interest Rate Decision", "USD"),
    ("CPI m/m", "USD"),
    ("Core CPI m/m", "USD"),
    ("Retail Sales m/m", "USD"),
    ("GDP q/q", "USD"),
    ("Unemployment Rate", "USD"),

    # EUR Events
    ("ECB Interest Rate Decision", "EUR"),
    ("ECB Press Conference", "EUR"),
    ("German CPI m/m", "EUR"),

    # GBP Events
    ("BOE Interest Rate Decision", "GBP"),
    ("UK CPI y/y", "GBP"),

    # JPY Events
    ("BOJ Policy Rate", "JPY"),
    ("BOJ Press Conference", "JPY"),
]
