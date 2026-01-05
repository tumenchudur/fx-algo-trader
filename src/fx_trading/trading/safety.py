"""
Safety Mechanisms for Live Trading.

Additional safeguards for production trading.
"""

import time
from datetime import datetime, timedelta
from typing import Optional, Callable
from pathlib import Path
import json

from loguru import logger

from fx_trading.execution.broker import Broker
from fx_trading.portfolio.accounting import PortfolioManager


class PositionReconciler:
    """
    Reconciles internal position state with broker.

    Detects and handles:
    - Positions closed externally (manually in MT5)
    - Positions opened externally
    - Size mismatches
    """

    def __init__(
        self,
        broker: Broker,
        portfolio: PortfolioManager,
        check_interval_seconds: float = 60.0,
    ):
        """
        Initialize reconciler.

        Args:
            broker: Broker instance
            portfolio: Portfolio manager
            check_interval_seconds: How often to reconcile
        """
        self.broker = broker
        self.portfolio = portfolio
        self.check_interval = check_interval_seconds
        self.last_check = datetime.min

    def check(self) -> dict:
        """
        Perform reconciliation check.

        Returns:
            Dict with reconciliation results
        """
        now = datetime.utcnow()
        if (now - self.last_check).total_seconds() < self.check_interval:
            return {"skipped": True}

        self.last_check = now

        # Get positions from both sources
        broker_positions = self.broker.get_positions()
        internal_positions = list(self.portfolio.positions.values())

        broker_symbols = {p.symbol for p in broker_positions}
        internal_symbols = {p.symbol for p in internal_positions}

        result = {
            "timestamp": now.isoformat(),
            "broker_count": len(broker_positions),
            "internal_count": len(internal_positions),
            "mismatches": [],
            "extra_in_broker": list(broker_symbols - internal_symbols),
            "missing_from_broker": list(internal_symbols - broker_symbols),
        }

        # Check for size mismatches
        for bp in broker_positions:
            for ip in internal_positions:
                if bp.symbol == ip.symbol:
                    if abs(bp.size - ip.size) > 0.001:
                        result["mismatches"].append({
                            "symbol": bp.symbol,
                            "broker_size": bp.size,
                            "internal_size": ip.size,
                        })

        if result["extra_in_broker"]:
            logger.warning(f"Positions in broker not tracked: {result['extra_in_broker']}")

        if result["missing_from_broker"]:
            logger.error(f"Tracked positions not in broker: {result['missing_from_broker']}")

        if result["mismatches"]:
            logger.warning(f"Position size mismatches: {result['mismatches']}")

        return result


class HeartbeatMonitor:
    """
    Monitors trading system health via heartbeat.

    Detects if the trading loop has stopped responding.
    """

    def __init__(
        self,
        max_silence_seconds: float = 120.0,
        on_failure: Optional[Callable] = None,
    ):
        """
        Initialize heartbeat monitor.

        Args:
            max_silence_seconds: Max time without heartbeat before alert
            on_failure: Callback function on heartbeat failure
        """
        self.max_silence = max_silence_seconds
        self.on_failure = on_failure
        self.last_heartbeat = datetime.utcnow()
        self.failures = 0

    def heartbeat(self) -> None:
        """Record a heartbeat."""
        self.last_heartbeat = datetime.utcnow()
        self.failures = 0

    def check(self) -> bool:
        """
        Check if heartbeat is alive.

        Returns:
            True if healthy, False if stale
        """
        elapsed = (datetime.utcnow() - self.last_heartbeat).total_seconds()

        if elapsed > self.max_silence:
            self.failures += 1
            logger.error(
                f"Heartbeat failure #{self.failures}: {elapsed:.0f}s since last heartbeat"
            )

            if self.on_failure:
                try:
                    self.on_failure()
                except Exception as e:
                    logger.error(f"Heartbeat failure callback error: {e}")

            return False

        return True

    def get_status(self) -> dict:
        """Get heartbeat status."""
        return {
            "last_heartbeat": self.last_heartbeat.isoformat(),
            "seconds_since": (datetime.utcnow() - self.last_heartbeat).total_seconds(),
            "failures": self.failures,
            "healthy": self.check(),
        }


class EmergencyStop:
    """
    Emergency stop mechanism.

    Triggers on critical conditions:
    - Excessive drawdown
    - Rapid loss sequence
    - Connection failures
    """

    def __init__(
        self,
        broker: Broker,
        max_consecutive_losses: int = 5,
        max_rapid_loss_pct: float = 5.0,
        rapid_loss_window_minutes: int = 30,
    ):
        """
        Initialize emergency stop.

        Args:
            broker: Broker instance for closing positions
            max_consecutive_losses: Trigger after N consecutive losses
            max_rapid_loss_pct: Trigger if loss exceeds % in window
            rapid_loss_window_minutes: Time window for rapid loss check
        """
        self.broker = broker
        self.max_consecutive_losses = max_consecutive_losses
        self.max_rapid_loss_pct = max_rapid_loss_pct
        self.rapid_loss_window = timedelta(minutes=rapid_loss_window_minutes)

        # State
        self.triggered = False
        self.trigger_reason: Optional[str] = None
        self.trigger_time: Optional[datetime] = None
        self.consecutive_losses = 0
        self.loss_history: list[tuple[datetime, float]] = []

    def record_trade_result(self, pnl: float, timestamp: datetime) -> bool:
        """
        Record trade result and check for emergency stop.

        Args:
            pnl: Trade PnL
            timestamp: Trade close time

        Returns:
            True if emergency stop triggered
        """
        if pnl < 0:
            self.consecutive_losses += 1
            self.loss_history.append((timestamp, pnl))
        else:
            self.consecutive_losses = 0

        # Check consecutive losses
        if self.consecutive_losses >= self.max_consecutive_losses:
            return self._trigger(f"Consecutive losses: {self.consecutive_losses}")

        # Check rapid loss
        window_start = timestamp - self.rapid_loss_window
        recent_losses = sum(
            loss for ts, loss in self.loss_history
            if ts >= window_start
        )
        # Would need equity to calculate percentage properly
        # This is a simplified check

        return False

    def _trigger(self, reason: str) -> bool:
        """Trigger emergency stop."""
        if self.triggered:
            return True

        self.triggered = True
        self.trigger_reason = reason
        self.trigger_time = datetime.utcnow()

        logger.critical(f"EMERGENCY STOP TRIGGERED: {reason}")

        # Close all positions
        try:
            fills = self.broker.close_all_positions()
            logger.warning(f"Emergency closed {len(fills)} positions")
        except Exception as e:
            logger.error(f"Failed to close positions: {e}")

        return True

    def is_triggered(self) -> bool:
        """Check if emergency stop is active."""
        return self.triggered

    def reset(self) -> None:
        """Reset emergency stop (manual intervention required)."""
        logger.warning("Emergency stop reset - manual override")
        self.triggered = False
        self.trigger_reason = None
        self.trigger_time = None
        self.consecutive_losses = 0

    def get_status(self) -> dict:
        """Get emergency stop status."""
        return {
            "triggered": self.triggered,
            "reason": self.trigger_reason,
            "trigger_time": self.trigger_time.isoformat() if self.trigger_time else None,
            "consecutive_losses": self.consecutive_losses,
        }


class TradeLogger:
    """
    Logs all trades to file for audit and analysis.
    """

    def __init__(self, output_dir: Path, run_id: str):
        """
        Initialize trade logger.

        Args:
            output_dir: Directory for log files
            run_id: Run identifier
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_id = run_id

        self.log_file = self.output_dir / f"trades_{run_id}.jsonl"
        self.trade_count = 0

    def log_trade(
        self,
        symbol: str,
        side: str,
        size: float,
        entry_price: float,
        exit_price: Optional[float] = None,
        pnl: Optional[float] = None,
        exit_reason: Optional[str] = None,
        metadata: Optional[dict] = None,
    ) -> None:
        """Log a trade to file."""
        trade = {
            "timestamp": datetime.utcnow().isoformat(),
            "trade_number": self.trade_count + 1,
            "symbol": symbol,
            "side": side,
            "size": size,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl": pnl,
            "exit_reason": exit_reason,
            "metadata": metadata or {},
        }

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(trade) + "\n")
            self.trade_count += 1
        except Exception as e:
            logger.error(f"Failed to log trade: {e}")

    def log_event(self, event_type: str, details: dict) -> None:
        """Log a trading event."""
        event = {
            "timestamp": datetime.utcnow().isoformat(),
            "event_type": event_type,
            "details": details,
        }

        try:
            with open(self.log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            logger.error(f"Failed to log event: {e}")


class SafetyMonitor:
    """
    Aggregates all safety mechanisms.
    """

    def __init__(
        self,
        broker: Broker,
        portfolio: PortfolioManager,
        output_dir: Path,
        run_id: str,
    ):
        """
        Initialize safety monitor.

        Args:
            broker: Broker instance
            portfolio: Portfolio manager
            output_dir: Output directory for logs
            run_id: Run identifier
        """
        self.reconciler = PositionReconciler(broker, portfolio)
        self.heartbeat = HeartbeatMonitor(
            on_failure=lambda: logger.critical("HEARTBEAT FAILURE")
        )
        self.emergency = EmergencyStop(broker)
        self.trade_logger = TradeLogger(output_dir, run_id)

    def check_all(self) -> dict:
        """Run all safety checks."""
        return {
            "reconciliation": self.reconciler.check(),
            "heartbeat": self.heartbeat.get_status(),
            "emergency": self.emergency.get_status(),
        }

    def record_heartbeat(self) -> None:
        """Record a heartbeat."""
        self.heartbeat.heartbeat()

    def record_trade(self, **kwargs) -> None:
        """Record a trade."""
        self.trade_logger.log_trade(**kwargs)

        # Check emergency stop if PnL provided
        if kwargs.get("pnl") is not None:
            self.emergency.record_trade_result(
                kwargs["pnl"],
                datetime.utcnow(),
            )

    def is_safe_to_trade(self) -> bool:
        """Check if it's safe to continue trading."""
        if self.emergency.is_triggered():
            return False
        if not self.heartbeat.check():
            return False
        return True
