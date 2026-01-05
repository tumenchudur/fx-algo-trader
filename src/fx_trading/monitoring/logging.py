"""
Structured logging for the trading system.

Provides consistent, parseable log output.
"""

import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from loguru import logger


def setup_logging(
    log_dir: Optional[Path] = None,
    level: str = "INFO",
    json_output: bool = False,
    run_id: Optional[str] = None,
) -> None:
    """
    Configure logging for the trading system.

    Args:
        log_dir: Directory for log files (None = console only)
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_output: Use JSON format for structured logging
        run_id: Run identifier for log files
    """
    # Remove default handler
    logger.remove()

    # Console format
    console_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> | "
        "<level>{message}</level>"
    )

    # Add console handler
    logger.add(
        sys.stderr,
        format=console_format,
        level=level,
        colorize=True,
    )

    # Add file handlers if log_dir specified
    if log_dir:
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)

        run_suffix = f"_{run_id}" if run_id else ""

        if json_output:
            # JSON format for structured logs
            logger.add(
                log_dir / f"trading{run_suffix}.json",
                format="{message}",
                level=level,
                serialize=True,
                rotation="10 MB",
                retention="7 days",
            )
        else:
            # Text format
            logger.add(
                log_dir / f"trading{run_suffix}.log",
                format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
                level=level,
                rotation="10 MB",
                retention="7 days",
            )

        # Separate error log
        logger.add(
            log_dir / f"errors{run_suffix}.log",
            format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} | {message}",
            level="ERROR",
            rotation="10 MB",
            retention="30 days",
        )

    logger.info(f"Logging configured: level={level}, json={json_output}")


class TradingLogger:
    """
    Structured logging for trading events.

    Provides methods for logging specific trading events with consistent format.
    """

    def __init__(self, run_id: str):
        """
        Initialize trading logger.

        Args:
            run_id: Run identifier
        """
        self.run_id = run_id
        self.events: list[dict] = []

    def log_signal(
        self,
        timestamp: datetime,
        symbol: str,
        side: str,
        strength: float,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log a trading signal."""
        event = {
            "event_type": "signal",
            "run_id": self.run_id,
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "side": side,
            "strength": strength,
            "stop_loss": stop_loss,
            "take_profit": take_profit,
            "metadata": metadata or {},
        }
        self.events.append(event)
        logger.info(f"SIGNAL: {symbol} {side} strength={strength:.2f}")

    def log_risk_decision(
        self,
        timestamp: datetime,
        symbol: str,
        approved: bool,
        reasons: list[str],
        adjusted_size: Optional[float] = None,
    ) -> None:
        """Log a risk decision."""
        event = {
            "event_type": "risk_decision",
            "run_id": self.run_id,
            "timestamp": timestamp.isoformat(),
            "symbol": symbol,
            "approved": approved,
            "reasons": reasons,
            "adjusted_size": adjusted_size,
        }
        self.events.append(event)

        if approved:
            logger.info(f"RISK: {symbol} APPROVED size={adjusted_size}")
        else:
            logger.warning(f"RISK: {symbol} REJECTED - {reasons}")

    def log_order(
        self,
        timestamp: datetime,
        order_id: str,
        symbol: str,
        side: str,
        order_type: str,
        size: float,
        price: Optional[float] = None,
    ) -> None:
        """Log an order."""
        event = {
            "event_type": "order",
            "run_id": self.run_id,
            "timestamp": timestamp.isoformat(),
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "order_type": order_type,
            "size": size,
            "price": price,
        }
        self.events.append(event)
        logger.info(f"ORDER: {order_id[:8]} {symbol} {side} {order_type} size={size}")

    def log_fill(
        self,
        timestamp: datetime,
        order_id: str,
        symbol: str,
        side: str,
        size: float,
        fill_price: float,
        commission: float,
        slippage: float,
    ) -> None:
        """Log an order fill."""
        event = {
            "event_type": "fill",
            "run_id": self.run_id,
            "timestamp": timestamp.isoformat(),
            "order_id": order_id,
            "symbol": symbol,
            "side": side,
            "size": size,
            "fill_price": fill_price,
            "commission": commission,
            "slippage": slippage,
        }
        self.events.append(event)
        logger.info(
            f"FILL: {order_id[:8]} {symbol} {side} size={size} "
            f"@ {fill_price:.5f} comm={commission:.2f}"
        )

    def log_trade_close(
        self,
        timestamp: datetime,
        trade_id: str,
        symbol: str,
        side: str,
        pnl: float,
        exit_reason: str,
    ) -> None:
        """Log a trade close."""
        event = {
            "event_type": "trade_close",
            "run_id": self.run_id,
            "timestamp": timestamp.isoformat(),
            "trade_id": trade_id,
            "symbol": symbol,
            "side": side,
            "pnl": pnl,
            "exit_reason": exit_reason,
        }
        self.events.append(event)

        if pnl >= 0:
            logger.info(f"TRADE: {trade_id[:8]} {symbol} {side} PnL=+{pnl:.2f} ({exit_reason})")
        else:
            logger.info(f"TRADE: {trade_id[:8]} {symbol} {side} PnL={pnl:.2f} ({exit_reason})")

    def log_daily_summary(
        self,
        date: datetime,
        equity: float,
        daily_pnl: float,
        trades: int,
        win_rate: float,
    ) -> None:
        """Log daily summary."""
        event = {
            "event_type": "daily_summary",
            "run_id": self.run_id,
            "date": date.strftime("%Y-%m-%d"),
            "equity": equity,
            "daily_pnl": daily_pnl,
            "trades": trades,
            "win_rate": win_rate,
        }
        self.events.append(event)
        logger.info(
            f"DAILY: {date.strftime('%Y-%m-%d')} equity={equity:.2f} "
            f"pnl={daily_pnl:+.2f} trades={trades} win_rate={win_rate:.1%}"
        )

    def log_kill_switch(
        self,
        timestamp: datetime,
        reason: str,
    ) -> None:
        """Log kill switch activation."""
        event = {
            "event_type": "kill_switch",
            "run_id": self.run_id,
            "timestamp": timestamp.isoformat(),
            "reason": reason,
        }
        self.events.append(event)
        logger.critical(f"KILL SWITCH: {reason}")

    def save_events(self, path: Path) -> None:
        """Save all events to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w") as f:
            json.dump(self.events, f, indent=2, default=str)

        logger.info(f"Saved {len(self.events)} events to {path}")

    def get_events_df(self) -> "pd.DataFrame":
        """Get events as DataFrame."""
        import pandas as pd
        return pd.DataFrame(self.events)
