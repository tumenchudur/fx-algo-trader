"""Telegram notification service for trade alerts.

Sends alerts when trades are executed, positions closed, or errors occur.

Setup:
1. Create a bot with @BotFather on Telegram
2. Get your chat ID by messaging @userinfobot
3. Add bot token and chat_id to your config
"""

import asyncio
from typing import Optional

import httpx
from loguru import logger


class TelegramNotifier:
    """Sends trade alerts via Telegram bot."""

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(
        self,
        bot_token: str,
        chat_id: str,
        enabled: bool = True,
    ):
        """
        Initialize Telegram notifier.

        Args:
            bot_token: Telegram bot token from @BotFather
            chat_id: Your Telegram chat ID
            enabled: Enable/disable notifications
        """
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.enabled = enabled
        self._client = httpx.Client(timeout=10.0)

    def send_message(self, message: str, parse_mode: str = "HTML") -> bool:
        """
        Send a message to Telegram.

        Args:
            message: Message text (supports HTML formatting)
            parse_mode: "HTML" or "Markdown"

        Returns:
            True if sent successfully
        """
        if not self.enabled:
            return True

        if not self.bot_token or not self.chat_id:
            logger.warning("Telegram not configured (missing token or chat_id)")
            return False

        url = f"{self.BASE_URL.format(token=self.bot_token)}/sendMessage"

        try:
            response = self._client.post(
                url,
                json={
                    "chat_id": self.chat_id,
                    "text": message,
                    "parse_mode": parse_mode,
                },
            )
            response.raise_for_status()
            return True
        except Exception as e:
            logger.error(f"Telegram send failed: {e}")
            return False

    def trade_opened(
        self,
        symbol: str,
        side: str,
        size: float,
        price: float,
        sl: Optional[float] = None,
        tp: Optional[float] = None,
    ) -> bool:
        """Send trade opened alert."""
        emoji = "\U0001F7E2" if side.upper() == "BUY" else "\U0001F534"  # Green/Red circle

        message = (
            f"{emoji} <b>TRADE OPENED</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Size:</b> {size} lots\n"
            f"<b>Price:</b> {price}"
        )

        if sl:
            message += f"\n<b>Stop Loss:</b> {sl}"
        if tp:
            message += f"\n<b>Take Profit:</b> {tp}"

        return self.send_message(message)

    def trade_closed(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        exit_price: float,
        profit: float,
        pips: Optional[float] = None,
    ) -> bool:
        """Send trade closed alert."""
        emoji = "\U00002705" if profit >= 0 else "\U0000274C"  # Checkmark / X
        profit_emoji = "\U0001F4B0" if profit >= 0 else "\U0001F4B8"  # Money bag / flying money

        message = (
            f"{emoji} <b>TRADE CLOSED</b>\n\n"
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Size:</b> {size} lots\n"
            f"<b>Entry:</b> {entry_price}\n"
            f"<b>Exit:</b> {exit_price}\n"
            f"{profit_emoji} <b>P/L:</b> ${profit:+.2f}"
        )

        if pips is not None:
            message += f" ({pips:+.1f} pips)"

        return self.send_message(message)

    def trade_rejected(
        self,
        symbol: str,
        side: str,
        reason: str,
    ) -> bool:
        """Send trade rejected alert."""
        message = (
            f"\U000026A0 <b>TRADE REJECTED</b>\n\n"  # Warning sign
            f"<b>Symbol:</b> {symbol}\n"
            f"<b>Side:</b> {side.upper()}\n"
            f"<b>Reason:</b> {reason}"
        )
        return self.send_message(message)

    def error_alert(self, error_message: str) -> bool:
        """Send error alert."""
        message = (
            f"\U0001F6A8 <b>ERROR</b>\n\n"  # Rotating light
            f"{error_message}"
        )
        return self.send_message(message)

    def kill_switch_triggered(
        self,
        reason: str,
        drawdown: float,
        equity: float,
    ) -> bool:
        """Send kill switch alert."""
        message = (
            f"\U0001F6D1 <b>KILL SWITCH TRIGGERED</b>\n\n"  # Stop sign
            f"<b>Reason:</b> {reason}\n"
            f"<b>Drawdown:</b> {drawdown:.2f}%\n"
            f"<b>Equity:</b> ${equity:.2f}\n\n"
            f"Trading has been stopped!"
        )
        return self.send_message(message)

    def daily_summary(
        self,
        trades: int,
        winners: int,
        losers: int,
        total_pnl: float,
        equity: float,
    ) -> bool:
        """Send daily summary."""
        win_rate = (winners / trades * 100) if trades > 0 else 0
        emoji = "\U0001F4C8" if total_pnl >= 0 else "\U0001F4C9"  # Chart up/down

        message = (
            f"{emoji} <b>DAILY SUMMARY</b>\n\n"
            f"<b>Trades:</b> {trades}\n"
            f"<b>Winners:</b> {winners} ({win_rate:.1f}%)\n"
            f"<b>Losers:</b> {losers}\n"
            f"<b>P/L:</b> ${total_pnl:+.2f}\n"
            f"<b>Equity:</b> ${equity:.2f}"
        )
        return self.send_message(message)

    def startup_message(
        self,
        symbols: list[str],
        strategy: str,
        equity: float,
    ) -> bool:
        """Send startup notification."""
        message = (
            f"\U0001F680 <b>TRADING STARTED</b>\n\n"  # Rocket
            f"<b>Strategy:</b> {strategy}\n"
            f"<b>Symbols:</b> {', '.join(symbols)}\n"
            f"<b>Equity:</b> ${equity:.2f}"
        )
        return self.send_message(message)

    def shutdown_message(self, reason: str = "Manual stop") -> bool:
        """Send shutdown notification."""
        message = (
            f"\U0001F6D1 <b>TRADING STOPPED</b>\n\n"  # Stop sign
            f"<b>Reason:</b> {reason}"
        )
        return self.send_message(message)

    def close(self):
        """Close HTTP client."""
        self._client.close()
