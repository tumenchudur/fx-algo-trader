"""
Live Trading Runner.

Main loop for running strategies against live/demo brokers.
"""

import signal
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from fx_trading.config.models import LiveTradingConfig
from fx_trading.execution.broker import Broker
from fx_trading.notifications.telegram import TelegramNotifier
from fx_trading.portfolio.accounting import PortfolioManager
from fx_trading.risk.engine import RiskEngine
from fx_trading.risk.news_filter import NewsFilter, NewsFilterConfig, TradingSchedule, Impact
from fx_trading.strategies.base import Strategy
from fx_trading.types.models import Order, OrderType, Side, PriceData


class LiveTradingRunner:
    """
    Runs strategies against a live/demo broker.

    Main trading loop:
    1. Fetch prices
    2. Update positions
    3. Check exits (SL/TP)
    4. Generate signals
    5. Evaluate risk
    6. Execute orders
    7. Log metrics
    8. Sleep
    """

    def __init__(
        self,
        config: LiveTradingConfig,
        broker: Broker,
        strategy: Strategy,
        risk_engine: RiskEngine,
        portfolio: PortfolioManager,
    ):
        """
        Initialize live trading runner.

        Args:
            config: Live trading configuration
            broker: Connected broker instance
            strategy: Strategy to run
            risk_engine: Risk engine for trade approval
            portfolio: Portfolio manager for position tracking
        """
        self.config = config
        self.broker = broker
        self.strategy = strategy
        self.risk_engine = risk_engine
        self.portfolio = portfolio

        # State
        self.running = False
        self.start_time: Optional[datetime] = None
        self.iteration = 0
        self.last_heartbeat = datetime.utcnow()

        # Data for strategy (rolling window)
        self.price_history: dict[str, list[dict]] = {s: [] for s in config.symbols}
        self.max_history_bars = 500  # Keep last N bars

        # Metrics
        self.signals_generated = 0
        self.trades_executed = 0
        self.trades_rejected = 0
        self.trades_blocked_by_news = 0

        # Initialize news filter
        self.news_filter = self._init_news_filter()
        self.trading_schedule = TradingSchedule(news_filter=self.news_filter)

        # Initialize Telegram notifier
        self.telegram = self._init_telegram()

        # Setup signal handlers for graceful shutdown
        self._setup_signal_handlers()

        # Create output directory
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"Live runner initialized: symbols={config.symbols}, "
            f"poll_interval={config.poll_interval_seconds}s"
        )

    def _init_news_filter(self) -> Optional[NewsFilter]:
        """Initialize news filter from config."""
        news_config = self.config.news_filter
        if not news_config.enabled:
            logger.info("News filter disabled")
            return None

        # Map config impact string to Impact enum
        impact_map = {
            "low": Impact.LOW,
            "medium": Impact.MEDIUM,
            "high": Impact.HIGH,
        }

        filter_config = NewsFilterConfig(
            enabled=True,
            minutes_before=news_config.minutes_before,
            minutes_after=news_config.minutes_after,
            min_impact=impact_map.get(news_config.min_impact, Impact.HIGH),
            currencies=news_config.currencies,
            block_modifications=news_config.block_modifications,
            finnhub_api_key=news_config.finnhub_api_key,
        )

        logger.info(
            f"News filter enabled: block {news_config.minutes_before}min before, "
            f"{news_config.minutes_after}min after {news_config.min_impact}-impact events"
        )

        return NewsFilter(filter_config)

    def _init_telegram(self) -> Optional[TelegramNotifier]:
        """Initialize Telegram notifier from config."""
        tg_config = self.config.telegram
        if not tg_config.enabled:
            logger.info("Telegram notifications disabled")
            return None

        if not tg_config.bot_token or not tg_config.chat_id:
            logger.warning("Telegram enabled but missing bot_token or chat_id")
            return None

        notifier = TelegramNotifier(
            bot_token=tg_config.bot_token,
            chat_id=tg_config.chat_id,
            enabled=True,
        )
        logger.info("Telegram notifications enabled")
        return notifier

    def _setup_signal_handlers(self) -> None:
        """Setup handlers for graceful shutdown."""
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)

    def _handle_shutdown(self, signum: int, frame) -> None:
        """Handle shutdown signal."""
        logger.warning(f"Shutdown signal received ({signum})")
        self.stop()

    def run(self) -> None:
        """
        Main trading loop.

        Runs until stopped or max runtime reached.
        """
        self.running = True
        self.start_time = datetime.utcnow()

        logger.info("=" * 60)
        logger.info("LIVE TRADING STARTED")
        logger.info(f"Symbols: {self.config.symbols}")
        logger.info(f"Strategy: {self.strategy.config.name}")
        logger.info("=" * 60)

        # Send Telegram startup notification
        if self.telegram:
            account = self.broker.get_account()
            self.telegram.startup_message(
                symbols=self.config.symbols,
                strategy=self.strategy.config.name,
                equity=account.equity if account else 0,
            )

        try:
            while self.running:
                self._run_iteration()
                self.iteration += 1

                # Check max runtime
                if self._check_max_runtime():
                    logger.info("Max runtime reached, stopping")
                    break

                # Sleep between iterations
                time.sleep(self.config.poll_interval_seconds)

        except Exception as e:
            logger.exception(f"Trading loop error: {e}")
            raise
        finally:
            self._shutdown()

    def _run_iteration(self) -> None:
        """Run one iteration of the trading loop."""
        current_time = datetime.utcnow()

        # Heartbeat logging
        if (current_time - self.last_heartbeat).total_seconds() >= self.config.heartbeat_interval_seconds:
            self._log_heartbeat()
            self.last_heartbeat = current_time

        # Process each symbol
        for symbol in self.config.symbols:
            try:
                self._process_symbol(symbol, current_time)
            except Exception as e:
                logger.error(f"Error processing {symbol}: {e}")

        # Update portfolio state
        self._update_portfolio()

    def _process_symbol(self, symbol: str, current_time: datetime) -> None:
        """Process a single symbol."""
        # Get current prices
        price_data = self.broker.get_prices(symbol)
        if not price_data:
            logger.debug(f"No price data for {symbol}")
            return

        # Store price in history
        self._store_price(symbol, price_data)

        # Check existing positions
        positions = self.broker.get_positions(symbol)
        for position in positions:
            self._check_position_exits(position, price_data)

        # Skip signal generation if we have a position
        if positions:
            return

        # Check news filter before generating signals
        schedule_result = self.trading_schedule.can_trade(symbol, current_time)
        if not schedule_result.allowed:
            if self.iteration % 60 == 0:  # Log every 60 iterations to avoid spam
                logger.info(f"Trading paused for {symbol}: {schedule_result.reason}")
                if schedule_result.resume_time:
                    logger.info(f"  Resume at: {schedule_result.resume_time.strftime('%H:%M:%S UTC')}")
            return

        # Generate signals from strategy
        df = self._get_price_dataframe(symbol)
        if df is None or len(df) < self.strategy.get_lookback_period():
            return

        signals = self.strategy.generate_signals(df, len(df) - 1)

        for signal in signals:
            if signal.side == Side.FLAT:
                continue

            self.signals_generated += 1
            logger.info(f"Signal: {symbol} {signal.side.value} SL={signal.stop_loss}")

            # Evaluate through risk engine
            decision = self.risk_engine.evaluate_signal(
                signal=signal,
                price_data=price_data,
                current_time=current_time,
            )

            if not decision.approved:
                self.trades_rejected += 1
                logger.warning(f"Trade rejected: {decision.get_rejection_reasons()}")
                continue

            # Create and execute order
            order = Order(
                symbol=symbol,
                side=signal.side,
                order_type=OrderType.MARKET,
                size=decision.adjusted_size,
                stop_loss=signal.stop_loss,
                take_profit=signal.take_profit,
            )

            fill = self.broker.place_order(order)
            if fill:
                self.trades_executed += 1
                logger.info(
                    f"Order filled: {symbol} {signal.side.value} "
                    f"size={order.size} @ {fill.fill_price}"
                )
                # Send Telegram notification
                if self.telegram and self.config.telegram.notify_on_trade:
                    self.telegram.trade_opened(
                        symbol=symbol,
                        side=signal.side.value,
                        size=order.size,
                        price=fill.fill_price,
                        sl=order.stop_loss,
                        tp=order.take_profit,
                    )

    def _check_position_exits(self, position, price_data: PriceData) -> None:
        """Check if position should be closed."""
        # Update PnL
        position.update_pnl(price_data.bid, price_data.ask)

        # Check stop loss
        if position.check_stop_loss(price_data.bid, price_data.ask):
            logger.info(f"Stop loss triggered for {position.symbol}")
            exit_price = price_data.bid if position.side.value == "BUY" else price_data.ask
            self.broker.close_position(position.id)
            # Send Telegram notification
            if self.telegram and self.config.telegram.notify_on_close:
                self.telegram.trade_closed(
                    symbol=position.symbol,
                    side=position.side.value,
                    size=position.size,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    profit=position.unrealized_pnl,
                )
            return

        # Check take profit
        if position.check_take_profit(price_data.bid, price_data.ask):
            logger.info(f"Take profit triggered for {position.symbol}")
            exit_price = price_data.bid if position.side.value == "BUY" else price_data.ask
            self.broker.close_position(position.id)
            # Send Telegram notification
            if self.telegram and self.config.telegram.notify_on_close:
                self.telegram.trade_closed(
                    symbol=position.symbol,
                    side=position.side.value,
                    size=position.size,
                    entry_price=position.entry_price,
                    exit_price=exit_price,
                    profit=position.unrealized_pnl,
                )
            return

    def _store_price(self, symbol: str, price_data: PriceData) -> None:
        """Store price in rolling history."""
        bar = {
            "datetime": price_data.timestamp,
            "open": price_data.mid,
            "high": price_data.ask,
            "low": price_data.bid,
            "close": price_data.mid,
            "bid": price_data.bid,
            "ask": price_data.ask,
            "volume": 0,
            "symbol": symbol,
        }
        self.price_history[symbol].append(bar)

        # Trim to max history
        if len(self.price_history[symbol]) > self.max_history_bars:
            self.price_history[symbol] = self.price_history[symbol][-self.max_history_bars:]

    def _get_price_dataframe(self, symbol: str) -> Optional[pd.DataFrame]:
        """Convert price history to DataFrame for strategy."""
        history = self.price_history.get(symbol, [])
        if not history:
            return None

        df = pd.DataFrame(history)
        df.set_index("datetime", inplace=True)
        return df

    def _update_portfolio(self) -> None:
        """Update portfolio state from broker."""
        account = self.broker.get_account()
        self.portfolio.account.balance = account.balance
        self.portfolio.account.update(account.unrealized_pnl, account.margin_used)

    def _check_max_runtime(self) -> bool:
        """Check if max runtime exceeded."""
        if not self.config.max_runtime_hours:
            return False

        elapsed = datetime.utcnow() - self.start_time
        max_runtime = timedelta(hours=self.config.max_runtime_hours)
        return elapsed >= max_runtime

    def _log_heartbeat(self) -> None:
        """Log periodic heartbeat with status."""
        account = self.broker.get_account()
        positions = self.broker.get_positions()

        logger.info("-" * 40)
        logger.info(f"HEARTBEAT - Iteration {self.iteration}")
        logger.info(f"Balance: ${account.balance:.2f}, Equity: ${account.equity:.2f}")
        logger.info(f"Positions: {len(positions)}, Drawdown: {account.drawdown_pct:.2f}%")
        logger.info(f"Signals: {self.signals_generated}, Executed: {self.trades_executed}, Rejected: {self.trades_rejected}")
        logger.info("-" * 40)

    def stop(self) -> None:
        """Stop the trading loop."""
        logger.info("Stopping trading loop...")
        self.running = False

    def _shutdown(self) -> None:
        """Shutdown procedure."""
        logger.info("=" * 60)
        logger.info("SHUTTING DOWN")

        # Send Telegram shutdown notification
        if self.telegram:
            self.telegram.shutdown_message("Trading stopped")
            self.telegram.close()

        # Close positions if configured
        if self.config.close_positions_on_exit:
            logger.warning("Closing all positions on exit...")
            fills = self.broker.close_all_positions()
            logger.info(f"Closed {len(fills)} positions")

        # Save final state
        self._save_final_report()

        # Disconnect broker
        self.broker.disconnect()

        logger.info("Shutdown complete")
        logger.info("=" * 60)

    def _save_final_report(self) -> None:
        """Save final trading report."""
        try:
            report = {
                "run_id": self.config.run_id,
                "start_time": self.start_time.isoformat() if self.start_time else None,
                "end_time": datetime.utcnow().isoformat(),
                "iterations": self.iteration,
                "signals_generated": self.signals_generated,
                "trades_executed": self.trades_executed,
                "trades_rejected": self.trades_rejected,
                "final_balance": self.portfolio.account.balance,
                "final_equity": self.portfolio.account.equity,
            }

            import json
            report_path = self.output_dir / f"live_report_{self.config.run_id}.json"
            with open(report_path, "w") as f:
                json.dump(report, f, indent=2)

            logger.info(f"Report saved: {report_path}")

        except Exception as e:
            logger.error(f"Failed to save report: {e}")
