"""Economic calendar data fetcher.

Fetches high-impact economic events to filter trading around news releases.
Supports multiple data sources with fallback.
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
    """Fetches and caches economic calendar events.

    Supports multiple data sources:
    - Finnhub (free API, recommended)
    - Forex Factory (often blocks requests)
    - Manual fallback for major events

    Get a free Finnhub API key at: https://finnhub.io/
    """

    FINNHUB_URL = "https://finnhub.io/api/v1/calendar/economic"
    FOREX_FACTORY_URL = "https://www.forexfactory.com/ffcal_week_this.xml"
    CACHE_DURATION = timedelta(hours=4)

    def __init__(self, finnhub_api_key: Optional[str] = None):
        """
        Initialize calendar.

        Args:
            finnhub_api_key: Optional Finnhub API key for better data.
                            Get free key at https://finnhub.io/
        """
        self._events: list[EconomicEvent] = []
        self._last_fetch: Optional[datetime] = None
        self._finnhub_api_key = finnhub_api_key
        self._http_client = httpx.Client(
            timeout=10.0,
            headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
        )

    def fetch_events(self, force: bool = False) -> list[EconomicEvent]:
        """Fetch economic events from available sources.

        Tries sources in order:
        1. Finnhub (if API key provided)
        2. Forex Factory
        3. Manual fallback

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

        events = []

        # Try Finnhub first (most reliable)
        if self._finnhub_api_key:
            try:
                events = self._fetch_finnhub()
                logger.info(f"Fetched {len(events)} events from Finnhub")
            except Exception as e:
                logger.warning(f"Finnhub fetch failed: {e}")

        # Try Forex Factory as backup
        if not events:
            try:
                events = self._fetch_forex_factory()
                logger.info(f"Fetched {len(events)} events from Forex Factory")
            except Exception as e:
                logger.debug(f"Forex Factory fetch failed: {e}")

        # Use manual fallback if all else fails
        if not events:
            events = self._get_manual_events()
            logger.info("Using manual event fallback")

        if events:
            self._events = events
            self._last_fetch = now

        return self._events

    def _fetch_finnhub(self) -> list[EconomicEvent]:
        """Fetch events from Finnhub API."""
        if not self._finnhub_api_key:
            return []

        now = datetime.utcnow()
        from_date = (now - timedelta(days=1)).strftime("%Y-%m-%d")
        to_date = (now + timedelta(days=7)).strftime("%Y-%m-%d")

        response = self._http_client.get(
            self.FINNHUB_URL,
            params={
                "from": from_date,
                "to": to_date,
                "token": self._finnhub_api_key,
            }
        )
        response.raise_for_status()
        data = response.json()

        events = []
        for item in data.get("economicCalendar", []):
            try:
                event = self._parse_finnhub_event(item)
                if event:
                    events.append(event)
            except Exception as e:
                logger.debug(f"Failed to parse Finnhub event: {e}")

        return events

    def _parse_finnhub_event(self, item: dict) -> Optional[EconomicEvent]:
        """Parse event from Finnhub response."""
        # Map Finnhub impact (1=low, 2=medium, 3=high)
        impact_map = {1: Impact.LOW, 2: Impact.MEDIUM, 3: Impact.HIGH}
        impact = impact_map.get(item.get("impact", 1), Impact.LOW)

        # Parse timestamp
        time_str = item.get("time", "")
        if not time_str:
            return None

        try:
            timestamp = datetime.fromisoformat(time_str.replace("Z", "+00:00"))
            timestamp = timestamp.replace(tzinfo=None)  # Remove timezone for consistency
        except ValueError:
            return None

        # Get currency from country code
        country = item.get("country", "")
        currency_map = {
            "US": "USD", "EU": "EUR", "GB": "GBP", "JP": "JPY",
            "AU": "AUD", "CA": "CAD", "CH": "CHF", "NZ": "NZD",
            "CN": "CNY", "DE": "EUR", "FR": "EUR", "IT": "EUR",
        }
        currency = currency_map.get(country, country)

        return EconomicEvent(
            title=item.get("event", "Unknown"),
            currency=currency,
            timestamp=timestamp,
            impact=impact,
            forecast=str(item.get("estimate", "")) if item.get("estimate") else None,
            previous=str(item.get("prev", "")) if item.get("prev") else None,
        )

    def _fetch_forex_factory(self) -> list[EconomicEvent]:
        """Parse Forex Factory XML feed."""
        response = self._http_client.get(self.FOREX_FACTORY_URL)
        response.raise_for_status()

        events = []
        root = ET.fromstring(response.text)

        for event_elem in root.findall(".//event"):
            try:
                event = self._parse_ff_event(event_elem)
                if event:
                    events.append(event)
            except Exception as e:
                logger.debug(f"Failed to parse event: {e}")
                continue

        return events

    def _parse_ff_event(self, elem: ET.Element) -> Optional[EconomicEvent]:
        """Parse single event from Forex Factory XML element."""
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
            timestamp = timestamp + timedelta(hours=5)  # ET to UTC
        except ValueError:
            return None

        impact_map = {
            "high": Impact.HIGH,
            "medium": Impact.MEDIUM,
            "low": Impact.LOW,
            "holiday": Impact.HIGH,
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

    def _get_manual_events(self) -> list[EconomicEvent]:
        """Generate placeholder events for known recurring high-impact events.

        This is a fallback when API sources fail.
        Creates events for the current week based on typical schedules.
        """
        events = []
        now = datetime.utcnow()

        # Major recurring events (approximate schedules)
        # These are placeholders - actual timing varies
        weekly_events = [
            # US - typically released at specific times
            ("US Jobless Claims", "USD", Impact.MEDIUM, 3, 13, 30),  # Thursday 8:30 ET
        ]

        # First Friday of month events
        if now.day <= 7 and now.weekday() == 4:  # First Friday
            events.append(EconomicEvent(
                title="Non-Farm Payrolls",
                currency="USD",
                timestamp=now.replace(hour=13, minute=30, second=0),
                impact=Impact.HIGH,
            ))

        # Add weekly events
        for title, currency, impact, weekday, hour, minute in weekly_events:
            # Find next occurrence
            days_ahead = weekday - now.weekday()
            if days_ahead < 0:
                days_ahead += 7
            event_date = now + timedelta(days=days_ahead)
            event_time = event_date.replace(hour=hour, minute=minute, second=0)

            events.append(EconomicEvent(
                title=title,
                currency=currency,
                timestamp=event_time,
                impact=impact,
            ))

        return events

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


# Fallback: Known high-impact events for major currencies
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
