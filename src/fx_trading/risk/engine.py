"""
Risk Management Engine.

CRITICAL SAFETY COMPONENT: Implements all risk checks and kill switches.
This is the primary defense against catastrophic losses.
"""

from datetime import datetime, timedelta
from typing import Optional

from loguru import logger

from fx_trading.config.models import RiskConfig
from fx_trading.portfolio.accounting import PortfolioManager
from fx_trading.portfolio.position_sizing import PositionSizer
from fx_trading.types.models import (
    Signal,
    Side,
    RiskDecision,
    RiskCheckResult,
    AccountState,
    PriceData,
)


class RiskEngine:
    """
    Risk management engine with comprehensive guardrails.

    Performs pre-trade and portfolio-level risk checks.
    Implements kill switches for catastrophic scenarios.

    ALL TRADES MUST PASS THROUGH THIS ENGINE.
    """

    def __init__(
        self,
        config: RiskConfig,
        portfolio: PortfolioManager,
    ):
        """
        Initialize risk engine.

        Args:
            config: Risk configuration
            portfolio: Portfolio manager instance
        """
        self.config = config
        self.portfolio = portfolio
        self.position_sizer = PositionSizer(config)

        # Kill switch state
        self.kill_switch_active = False
        self.kill_switch_reason: Optional[str] = None
        self.kill_switch_time: Optional[datetime] = None

        # Anti-pattern tracking
        self.last_loss_time: Optional[datetime] = None
        self.consecutive_losses = 0
        self.last_trade_size: Optional[float] = None

        # Daily tracking
        self.daily_start_equity: Optional[float] = None
        self.current_date: Optional[datetime] = None

        logger.info(
            f"Risk engine initialized: max_risk={config.max_risk_per_trade_pct}%, "
            f"daily_loss_limit={config.daily_loss_limit_pct}%, "
            f"max_drawdown={config.max_drawdown_pct}%"
        )

    def evaluate_signal(
        self,
        signal: Signal,
        price_data: PriceData,
        current_time: datetime,
        volatility: Optional[float] = None,
    ) -> RiskDecision:
        """
        Evaluate a trading signal against all risk checks.

        Args:
            signal: Trading signal to evaluate
            price_data: Current price data
            current_time: Current timestamp
            volatility: Current ATR or volatility measure

        Returns:
            RiskDecision with approval status and details
        """
        decision = RiskDecision(
            timestamp=current_time,
            symbol=signal.symbol,
            approved=True,
        )

        # Check kill switch first
        if self.kill_switch_active:
            decision.approved = False
            decision.kill_switch_active = True
            decision.add_check(RiskCheckResult(
                check_name="kill_switch",
                passed=False,
                reason=f"Kill switch active: {self.kill_switch_reason}",
            ))
            logger.warning(f"Trade blocked: Kill switch active - {self.kill_switch_reason}")
            return decision

        # Skip checks for flat signals
        if signal.side == Side.FLAT:
            decision.add_check(RiskCheckResult(
                check_name="signal_type",
                passed=True,
                reason="Flat signal - no position change",
            ))
            return decision

        # Run all risk checks
        self._check_stale_price(price_data, current_time, decision)
        self._check_spread(price_data, decision)
        self._check_max_positions(signal, decision)
        self._check_daily_loss_limit(current_time, decision)
        self._check_max_drawdown(decision)
        self._check_exposure_limits(signal, price_data, decision)
        self._check_volatility(volatility, decision)
        self._check_anti_patterns(signal, current_time, decision)

        # Calculate position size if approved
        if decision.approved:
            size = self._calculate_risk_adjusted_size(
                signal=signal,
                price_data=price_data,
                volatility=volatility,
            )
            decision.adjusted_size = size
            decision.original_size = size

            if size <= 0:
                decision.approved = False
                decision.add_check(RiskCheckResult(
                    check_name="position_size",
                    passed=False,
                    reason="Calculated position size is zero or negative",
                ))

        # Log decision
        if decision.approved:
            logger.info(
                f"Trade approved: {signal.symbol} {signal.side.value} "
                f"size={decision.adjusted_size:.4f}"
            )
        else:
            reasons = decision.get_rejection_reasons()
            logger.warning(f"Trade rejected: {signal.symbol} - {reasons}")

        return decision

    def _check_stale_price(
        self,
        price_data: PriceData,
        current_time: datetime,
        decision: RiskDecision,
    ) -> None:
        """Check if price data is stale."""
        is_stale = price_data.is_stale(current_time, self.config.stale_price_seconds)

        if is_stale:
            # Calculate age safely handling timezone differences
            age = self._get_time_diff_seconds(current_time, price_data.timestamp)
            decision.add_check(RiskCheckResult(
                check_name="stale_price",
                passed=False,
                reason=f"Price data is {age:.0f}s old (max: {self.config.stale_price_seconds}s)",
                value=age,
                threshold=self.config.stale_price_seconds,
            ))
        else:
            decision.add_check(RiskCheckResult(
                check_name="stale_price",
                passed=True,
                reason="Price data is fresh",
            ))

    def _check_spread(
        self,
        price_data: PriceData,
        decision: RiskDecision,
    ) -> None:
        """Check if spread is within acceptable limits."""
        spread_pips = price_data.spread_pips

        if spread_pips > self.config.max_spread_pips:
            decision.add_check(RiskCheckResult(
                check_name="spread_filter",
                passed=False,
                reason=f"Spread {spread_pips:.1f} pips exceeds max {self.config.max_spread_pips} pips",
                value=spread_pips,
                threshold=self.config.max_spread_pips,
            ))
        else:
            decision.add_check(RiskCheckResult(
                check_name="spread_filter",
                passed=True,
                reason=f"Spread {spread_pips:.1f} pips within limit",
            ))

    def _check_max_positions(
        self,
        signal: Signal,
        decision: RiskDecision,
    ) -> None:
        """Check if max open positions limit is reached."""
        current_positions = len(self.portfolio.positions)

        # Check if we already have a position in this symbol
        symbol_positions = [
            p for p in self.portfolio.positions.values()
            if p.symbol == signal.symbol
        ]

        if len(symbol_positions) > 0:
            decision.add_check(RiskCheckResult(
                check_name="existing_position",
                passed=False,
                reason=f"Already have position in {signal.symbol}",
                value=len(symbol_positions),
                threshold=0,
            ))
            return

        if current_positions >= self.config.max_open_positions:
            decision.add_check(RiskCheckResult(
                check_name="max_positions",
                passed=False,
                reason=f"Max positions ({self.config.max_open_positions}) reached",
                value=current_positions,
                threshold=self.config.max_open_positions,
            ))
        else:
            decision.add_check(RiskCheckResult(
                check_name="max_positions",
                passed=True,
                reason=f"{current_positions}/{self.config.max_open_positions} positions",
            ))

    def _check_daily_loss_limit(
        self,
        current_time: datetime,
        decision: RiskDecision,
    ) -> None:
        """Check if daily loss limit has been reached."""
        # Reset daily tracking on new day
        current_date = current_time.date()
        if self.current_date != current_date:
            self.current_date = current_date
            self.daily_start_equity = self.portfolio.account.equity

        if self.daily_start_equity is None:
            self.daily_start_equity = self.portfolio.account.equity

        # Calculate daily loss
        daily_loss = self.daily_start_equity - self.portfolio.account.equity
        daily_loss_pct = (daily_loss / self.daily_start_equity) * 100 if self.daily_start_equity > 0 else 0

        if daily_loss_pct >= self.config.daily_loss_limit_pct:
            decision.add_check(RiskCheckResult(
                check_name="daily_loss_limit",
                passed=False,
                reason=f"Daily loss {daily_loss_pct:.2f}% exceeds limit {self.config.daily_loss_limit_pct}%",
                value=daily_loss_pct,
                threshold=self.config.daily_loss_limit_pct,
            ))
            # Trigger kill switch
            self._activate_kill_switch(
                f"Daily loss limit exceeded: {daily_loss_pct:.2f}%",
                current_time,
            )
        else:
            decision.add_check(RiskCheckResult(
                check_name="daily_loss_limit",
                passed=True,
                reason=f"Daily PnL: {-daily_loss_pct:.2f}%",
            ))

    def _check_max_drawdown(
        self,
        decision: RiskDecision,
    ) -> None:
        """Check if max drawdown has been reached."""
        drawdown_pct = self.portfolio.account.drawdown_pct

        if drawdown_pct >= self.config.max_drawdown_pct:
            decision.add_check(RiskCheckResult(
                check_name="max_drawdown",
                passed=False,
                reason=f"Drawdown {drawdown_pct:.2f}% exceeds max {self.config.max_drawdown_pct}%",
                value=drawdown_pct,
                threshold=self.config.max_drawdown_pct,
            ))
            # Trigger kill switch
            self._activate_kill_switch(
                f"Max drawdown exceeded: {drawdown_pct:.2f}%",
                datetime.utcnow(),
            )
        else:
            decision.add_check(RiskCheckResult(
                check_name="max_drawdown",
                passed=True,
                reason=f"Drawdown: {drawdown_pct:.2f}%",
            ))

    def _check_exposure_limits(
        self,
        signal: Signal,
        price_data: PriceData,
        decision: RiskDecision,
    ) -> None:
        """Check exposure limits (total and per-currency)."""
        equity = self.portfolio.account.equity
        current_exposure = self.portfolio.get_total_exposure()
        exposure_pct = (current_exposure / equity) * 100 if equity > 0 else 0

        if exposure_pct >= self.config.max_total_exposure_pct:
            decision.add_check(RiskCheckResult(
                check_name="total_exposure",
                passed=False,
                reason=f"Total exposure {exposure_pct:.1f}% at limit",
                value=exposure_pct,
                threshold=self.config.max_total_exposure_pct,
            ))
        else:
            decision.add_check(RiskCheckResult(
                check_name="total_exposure",
                passed=True,
                reason=f"Exposure: {exposure_pct:.1f}%",
            ))

        # Check per-currency exposure
        currency_exposure = self.portfolio.get_exposure_by_currency()
        base_currency = signal.symbol[:3]
        quote_currency = signal.symbol[3:]

        for currency in [base_currency, quote_currency]:
            if currency in currency_exposure:
                curr_exposure = abs(currency_exposure[currency])
                curr_exposure_pct = (curr_exposure / equity) * 100 if equity > 0 else 0

                if curr_exposure_pct >= self.config.max_exposure_per_currency_pct:
                    decision.add_check(RiskCheckResult(
                        check_name=f"currency_exposure_{currency}",
                        passed=False,
                        reason=f"{currency} exposure {curr_exposure_pct:.1f}% at limit",
                        value=curr_exposure_pct,
                        threshold=self.config.max_exposure_per_currency_pct,
                    ))

    def _check_volatility(
        self,
        volatility: Optional[float],
        decision: RiskDecision,
    ) -> None:
        """Check volatility filter if enabled."""
        if not self.config.enable_volatility_filter or volatility is None:
            decision.add_check(RiskCheckResult(
                check_name="volatility_filter",
                passed=True,
                reason="Volatility filter disabled or no data",
            ))
            return

        # This would require historical ATR average to be meaningful
        # For now, just check if volatility is present
        decision.add_check(RiskCheckResult(
            check_name="volatility_filter",
            passed=True,
            reason=f"Volatility: {volatility:.5f}",
        ))

    def _check_anti_patterns(
        self,
        signal: Signal,
        current_time: datetime,
        decision: RiskDecision,
    ) -> None:
        """Check for anti-patterns (martingale, revenge trading)."""
        # Check revenge trading (trading too soon after a loss)
        if self.config.no_revenge_trading and self.last_loss_time is not None:
            cooldown = timedelta(minutes=self.config.revenge_cooldown_bars * 5)  # Assume 5min bars
            time_since_loss_sec = self._get_time_diff_seconds(current_time, self.last_loss_time)

            if time_since_loss_sec < cooldown.total_seconds():
                decision.add_check(RiskCheckResult(
                    check_name="revenge_trading",
                    passed=False,
                    reason=f"Cooldown after loss: {time_since_loss_sec:.0f}s < {cooldown.total_seconds():.0f}s",
                ))
                return

        # Martingale check would require knowing proposed size vs last trade
        # This is checked in position sizing

        decision.add_check(RiskCheckResult(
            check_name="anti_patterns",
            passed=True,
            reason="No dangerous patterns detected",
        ))

    def _calculate_risk_adjusted_size(
        self,
        signal: Signal,
        price_data: PriceData,
        volatility: Optional[float],
    ) -> float:
        """Calculate risk-adjusted position size."""
        equity = self.portfolio.account.equity

        # Use strategy stop loss or volatility-based
        entry_price = price_data.ask if signal.side == Side.LONG else price_data.bid

        if signal.stop_loss is not None:
            size = self.position_sizer.calculate_size(
                equity=equity,
                entry_price=entry_price,
                stop_loss=signal.stop_loss,
                side=signal.side,
                symbol=signal.symbol,
            )
        elif volatility is not None:
            size = self.position_sizer.calculate_volatility_adjusted_size(
                equity=equity,
                atr=volatility,
                atr_multiplier=2.0,
                entry_price=entry_price,
                symbol=signal.symbol,
            )
        else:
            # Fallback to minimum size
            size = self.config.min_position_size_lots

        # Check exposure limits
        size, _ = self.position_sizer.check_exposure_limit(
            proposed_size=size,
            current_exposure=self.portfolio.get_total_exposure(),
            equity=equity,
            entry_price=entry_price,
        )

        # Check leverage limits
        size, _ = self.position_sizer.check_leverage_limit(
            proposed_size=size,
            current_exposure=self.portfolio.get_total_exposure(),
            equity=equity,
            entry_price=entry_price,
        )

        # Anti-martingale check
        if self.config.no_martingale and self.last_trade_size is not None:
            if size > self.last_trade_size * 1.5:  # More than 50% increase
                logger.warning(
                    f"Martingale pattern detected: size {size:.4f} > last {self.last_trade_size:.4f}*1.5"
                )
                size = self.last_trade_size

        return size

    def _activate_kill_switch(
        self,
        reason: str,
        timestamp: datetime,
    ) -> None:
        """Activate the kill switch."""
        if self.kill_switch_active:
            return

        self.kill_switch_active = True
        self.kill_switch_reason = reason
        self.kill_switch_time = timestamp

        logger.critical(f"KILL SWITCH ACTIVATED: {reason}")

        # Close all positions if configured
        if self.config.close_positions_on_kill:
            logger.warning("Closing all positions due to kill switch")
            # Note: Actual closing is handled by the execution layer

    def reset_kill_switch(self) -> None:
        """Reset kill switch (requires manual intervention)."""
        logger.warning("Kill switch reset - manual override")
        self.kill_switch_active = False
        self.kill_switch_reason = None
        self.kill_switch_time = None

    def record_trade_result(
        self,
        pnl: float,
        size: float,
        timestamp: datetime,
    ) -> None:
        """
        Record trade result for anti-pattern tracking.

        Args:
            pnl: Trade PnL
            size: Trade size
            timestamp: Trade completion time
        """
        self.last_trade_size = size

        if pnl < 0:
            self.consecutive_losses += 1
            self.last_loss_time = timestamp
            logger.debug(f"Loss recorded: {pnl:.2f}, consecutive: {self.consecutive_losses}")
        else:
            self.consecutive_losses = 0
            logger.debug(f"Win recorded: {pnl:.2f}")

    def get_status(self) -> dict:
        """Get current risk engine status."""
        return {
            "kill_switch_active": self.kill_switch_active,
            "kill_switch_reason": self.kill_switch_reason,
            "kill_switch_time": self.kill_switch_time.isoformat() if self.kill_switch_time else None,
            "consecutive_losses": self.consecutive_losses,
            "daily_start_equity": self.daily_start_equity,
            "current_drawdown_pct": self.portfolio.account.drawdown_pct,
            "open_positions": len(self.portfolio.positions),
            "total_exposure": self.portfolio.get_total_exposure(),
        }

    def should_close_all_positions(self) -> bool:
        """Check if all positions should be closed (kill switch + close config)."""
        return self.kill_switch_active and self.config.close_positions_on_kill

    def _get_time_diff_seconds(self, time1: datetime, time2: datetime) -> float:
        """
        Get time difference in seconds, handling timezone mismatches.

        Args:
            time1: First datetime (typically current time)
            time2: Second datetime (typically price timestamp)

        Returns:
            Absolute difference in seconds
        """
        t1 = time1
        t2 = time2

        # Convert pandas Timestamp to datetime if needed
        if hasattr(t1, 'to_pydatetime'):
            t1 = t1.to_pydatetime()
        if hasattr(t2, 'to_pydatetime'):
            t2 = t2.to_pydatetime()

        # Make both naive (assume UTC) to avoid comparison issues
        if hasattr(t1, 'tzinfo') and t1.tzinfo is not None:
            t1 = t1.replace(tzinfo=None)
        if hasattr(t2, 'tzinfo') and t2.tzinfo is not None:
            t2 = t2.replace(tzinfo=None)

        return abs((t1 - t2).total_seconds())
