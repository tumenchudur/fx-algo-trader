"""
Portfolio accounting and trade tracking.

Manages equity, positions, and realized/unrealized PnL.
"""

from dataclasses import dataclass, field
from datetime import datetime, date
from typing import Optional
from uuid import UUID

import pandas as pd
from loguru import logger

from fx_trading.types.models import (
    AccountState,
    Position,
    PositionStatus,
    Trade,
    Fill,
    Side,
    PortfolioSnapshot,
    DailySummary,
)


@dataclass
class TradeLog:
    """Maintains complete trade history."""

    trades: list[Trade] = field(default_factory=list)

    def add_trade(self, trade: Trade) -> None:
        """Add completed trade to log."""
        self.trades.append(trade)
        logger.info(
            f"Trade logged: {trade.symbol} {trade.side.value} "
            f"size={trade.size} pnl={trade.net_pnl:.2f}"
        )

    def get_trades_df(self) -> pd.DataFrame:
        """Convert trades to DataFrame."""
        if not self.trades:
            return pd.DataFrame()

        records = []
        for t in self.trades:
            records.append({
                "id": str(t.id),
                "symbol": t.symbol,
                "side": t.side.value,
                "size": t.size,
                "entry_time": t.entry_time,
                "exit_time": t.exit_time,
                "entry_price": t.entry_price,
                "exit_price": t.exit_price,
                "gross_pnl": t.gross_pnl,
                "net_pnl": t.net_pnl,
                "commission": t.total_commission,
                "slippage": t.total_slippage,
                "r_multiple": t.r_multiple,
                "exit_reason": t.exit_reason,
                "bars_held": t.bars_held,
            })

        return pd.DataFrame(records)

    def get_stats(self) -> dict:
        """Calculate trade statistics."""
        if not self.trades:
            return {
                "total_trades": 0,
                "winning_trades": 0,
                "losing_trades": 0,
                "win_rate": 0.0,
                "avg_win": 0.0,
                "avg_loss": 0.0,
                "profit_factor": 0.0,
                "expectancy": 0.0,
                "total_pnl": 0.0,
                "max_win": 0.0,
                "max_loss": 0.0,
            }

        wins = [t for t in self.trades if t.net_pnl > 0]
        losses = [t for t in self.trades if t.net_pnl <= 0]

        total_wins = sum(t.net_pnl for t in wins)
        total_losses = abs(sum(t.net_pnl for t in losses))

        avg_win = total_wins / len(wins) if wins else 0
        avg_loss = total_losses / len(losses) if losses else 0
        win_rate = len(wins) / len(self.trades) if self.trades else 0

        profit_factor = total_wins / total_losses if total_losses > 0 else float("inf")
        expectancy = (win_rate * avg_win) - ((1 - win_rate) * avg_loss)

        return {
            "total_trades": len(self.trades),
            "winning_trades": len(wins),
            "losing_trades": len(losses),
            "win_rate": win_rate,
            "avg_win": avg_win,
            "avg_loss": avg_loss,
            "profit_factor": profit_factor,
            "expectancy": expectancy,
            "total_pnl": sum(t.net_pnl for t in self.trades),
            "max_win": max((t.net_pnl for t in self.trades), default=0),
            "max_loss": min((t.net_pnl for t in self.trades), default=0),
        }


class PortfolioManager:
    """
    Manages portfolio state, positions, and accounting.

    Tracks equity curve, positions, and all PnL.
    """

    def __init__(
        self,
        initial_capital: float = 10000.0,
        base_currency: str = "USD",
    ):
        """
        Initialize portfolio manager.

        Args:
            initial_capital: Starting capital
            base_currency: Account currency
        """
        self.initial_capital = initial_capital
        self.base_currency = base_currency

        # Account state
        self.account = AccountState(
            balance=initial_capital,
            peak_equity=initial_capital,
        )

        # Position tracking
        self.positions: dict[UUID, Position] = {}

        # Trade history
        self.trade_log = TradeLog()

        # Equity curve
        self.equity_curve: list[PortfolioSnapshot] = []

        # Daily tracking
        self.daily_start_equity: float = initial_capital
        self.daily_pnl: float = 0.0
        self.current_date: Optional[date] = None
        self.daily_summaries: list[DailySummary] = []

    def open_position(
        self,
        symbol: str,
        side: Side,
        size: float,
        entry_price: float,
        entry_time: datetime,
        entry_bar_index: int,
        stop_loss: Optional[float] = None,
        take_profit: Optional[float] = None,
        time_exit_bars: Optional[int] = None,
        entry_commission: float = 0.0,
        entry_slippage: float = 0.0,
    ) -> Position:
        """
        Open a new position.

        Args:
            symbol: Trading symbol
            side: Position side
            size: Position size in lots
            entry_price: Entry price (after spread/slippage)
            entry_time: Entry timestamp
            entry_bar_index: Bar index at entry
            stop_loss: Stop loss price
            take_profit: Take profit price
            time_exit_bars: Exit after N bars
            entry_commission: Entry commission paid
            entry_slippage: Entry slippage paid

        Returns:
            New Position object
        """
        position = Position(
            symbol=symbol,
            side=side,
            size=size,
            entry_price=entry_price,
            entry_time=entry_time,
            current_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            entry_bar_index=entry_bar_index,
            time_exit_bars=time_exit_bars,
            status=PositionStatus.OPEN,
            entry_commission=entry_commission,
            entry_slippage=entry_slippage,
        )

        self.positions[position.id] = position

        # Calculate margin (simplified: 1% margin requirement)
        margin = size * 100000 * entry_price * 0.01
        self.account.margin_used += margin

        logger.info(
            f"Position opened: {symbol} {side.value} "
            f"size={size} @ {entry_price:.5f} "
            f"SL={stop_loss} TP={take_profit}"
        )

        return position

    def close_position(
        self,
        position_id: UUID,
        exit_price: float,
        exit_time: datetime,
        exit_bar_index: int,
        exit_commission: float = 0.0,
        exit_slippage: float = 0.0,
        exit_reason: str = "SIGNAL",
    ) -> Trade:
        """
        Close an existing position.

        Args:
            position_id: Position UUID
            exit_price: Exit price (after spread/slippage)
            exit_time: Exit timestamp
            exit_bar_index: Bar index at exit
            exit_commission: Exit commission paid
            exit_slippage: Exit slippage paid
            exit_reason: Reason for exit

        Returns:
            Completed Trade object
        """
        if position_id not in self.positions:
            raise ValueError(f"Position {position_id} not found")

        position = self.positions[position_id]
        position.status = PositionStatus.CLOSED

        # Create trade record
        trade = Trade(
            id=position.id,
            symbol=position.symbol,
            side=position.side,
            size=position.size,
            entry_time=position.entry_time,
            exit_time=exit_time,
            entry_price=position.entry_price,
            exit_price=exit_price,
            entry_commission=position.entry_commission,
            exit_commission=exit_commission,
            entry_slippage=position.entry_slippage,
            exit_slippage=exit_slippage,
            exit_reason=exit_reason,
            bars_held=exit_bar_index - position.entry_bar_index,
        )

        # Calculate PnL
        trade.calculate_pnl()

        # Calculate R-multiple if stop loss was set
        if position.stop_loss is not None:
            risk_per_unit = abs(position.entry_price - position.stop_loss)
            if risk_per_unit > 0:
                pnl_per_unit = (exit_price - position.entry_price) if position.side == Side.LONG else (position.entry_price - exit_price)
                trade.r_multiple = pnl_per_unit / risk_per_unit

        # Update account
        self.account.balance += trade.net_pnl
        self.account.realized_pnl += trade.net_pnl
        self.daily_pnl += trade.net_pnl

        # Release margin
        margin = position.size * 100000 * position.entry_price * 0.01
        self.account.margin_used -= margin

        # Log trade
        self.trade_log.add_trade(trade)

        # Remove from active positions
        del self.positions[position_id]

        logger.info(
            f"Position closed: {position.symbol} {position.side.value} "
            f"size={position.size} @ {exit_price:.5f} "
            f"pnl={trade.net_pnl:.2f} reason={exit_reason}"
        )

        return trade

    def update_positions(self, prices: dict[str, tuple[float, float]]) -> float:
        """
        Update all positions with current prices.

        Args:
            prices: Dict of symbol -> (bid, ask)

        Returns:
            Total unrealized PnL
        """
        total_unrealized = 0.0

        for position in self.positions.values():
            if position.symbol in prices:
                bid, ask = prices[position.symbol]
                position.update_pnl(bid, ask)
                total_unrealized += position.unrealized_pnl

        self.account.unrealized_pnl = total_unrealized
        self.account.update(total_unrealized, self.account.margin_used)

        return total_unrealized

    def record_snapshot(self, timestamp: datetime) -> PortfolioSnapshot:
        """
        Record current portfolio state.

        Args:
            timestamp: Current timestamp

        Returns:
            PortfolioSnapshot
        """
        snapshot = PortfolioSnapshot(
            timestamp=timestamp,
            equity=self.account.equity,
            balance=self.account.balance,
            unrealized_pnl=self.account.unrealized_pnl,
            realized_pnl=self.account.realized_pnl,
            drawdown_pct=self.account.drawdown_pct,
            num_positions=len(self.positions),
            total_exposure=self.get_total_exposure(),
            margin_used=self.account.margin_used,
        )

        self.equity_curve.append(snapshot)
        return snapshot

    def check_new_day(self, timestamp: datetime) -> Optional[DailySummary]:
        """
        Check for day change and record daily summary.

        Args:
            timestamp: Current timestamp

        Returns:
            DailySummary if day changed, None otherwise
        """
        current = timestamp.date()

        if self.current_date is None:
            self.current_date = current
            self.daily_start_equity = self.account.equity
            return None

        if current != self.current_date:
            # Day changed, create summary
            summary = self._create_daily_summary()
            self.daily_summaries.append(summary)

            # Reset for new day
            self.current_date = current
            self.daily_start_equity = self.account.equity
            self.daily_pnl = 0.0

            return summary

        return None

    def _create_daily_summary(self) -> DailySummary:
        """Create daily summary for current day."""
        # Get trades for today
        today_trades = [
            t for t in self.trade_log.trades
            if t.exit_time and t.exit_time.date() == self.current_date
        ]

        winning = [t for t in today_trades if t.net_pnl > 0]
        losing = [t for t in today_trades if t.net_pnl <= 0]

        daily_return = self.account.equity - self.daily_start_equity
        daily_return_pct = (daily_return / self.daily_start_equity) * 100 if self.daily_start_equity > 0 else 0

        return DailySummary(
            date=datetime.combine(self.current_date, datetime.min.time()),
            starting_equity=self.daily_start_equity,
            ending_equity=self.account.equity,
            daily_return=daily_return,
            daily_return_pct=daily_return_pct,
            num_trades=len(today_trades),
            winning_trades=len(winning),
            losing_trades=len(losing),
            gross_pnl=sum(t.gross_pnl for t in today_trades),
            net_pnl=sum(t.net_pnl for t in today_trades),
            total_commission=sum(t.total_commission for t in today_trades),
            total_slippage=sum(t.total_slippage for t in today_trades),
            max_drawdown_pct=self.account.drawdown_pct,
        )

    def get_total_exposure(self) -> float:
        """Get total position exposure in account currency."""
        return sum(
            p.size * 100000 * p.current_price
            for p in self.positions.values()
        )

    def get_exposure_by_currency(self) -> dict[str, float]:
        """Get exposure breakdown by currency."""
        exposure = {}

        for position in self.positions.values():
            # Extract currencies from symbol (e.g., EURUSD -> EUR, USD)
            base = position.symbol[:3]
            quote = position.symbol[3:]

            value = position.size * 100000 * position.current_price

            if position.side == Side.LONG:
                exposure[base] = exposure.get(base, 0) + value
                exposure[quote] = exposure.get(quote, 0) - value
            else:
                exposure[base] = exposure.get(base, 0) - value
                exposure[quote] = exposure.get(quote, 0) + value

        return exposure

    def get_open_positions(self, symbol: Optional[str] = None) -> list[Position]:
        """Get list of open positions, optionally filtered by symbol."""
        positions = list(self.positions.values())
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        return positions

    def get_equity_curve_df(self) -> pd.DataFrame:
        """Get equity curve as DataFrame."""
        if not self.equity_curve:
            return pd.DataFrame()

        records = [
            {
                "timestamp": s.timestamp,
                "equity": s.equity,
                "balance": s.balance,
                "unrealized_pnl": s.unrealized_pnl,
                "realized_pnl": s.realized_pnl,
                "drawdown_pct": s.drawdown_pct,
                "num_positions": s.num_positions,
                "total_exposure": s.total_exposure,
            }
            for s in self.equity_curve
        ]

        return pd.DataFrame(records).set_index("timestamp")

    def get_performance_metrics(self) -> dict:
        """Calculate overall performance metrics."""
        if not self.equity_curve:
            return {}

        equity_df = self.get_equity_curve_df()

        total_return = (self.account.equity - self.initial_capital) / self.initial_capital * 100
        max_drawdown = equity_df["drawdown_pct"].max() if len(equity_df) > 0 else 0

        # Calculate returns
        equity_df["returns"] = equity_df["equity"].pct_change()
        returns = equity_df["returns"].dropna()

        if len(returns) > 1:
            sharpe = (returns.mean() / returns.std()) * (252 ** 0.5) if returns.std() > 0 else 0
            sortino_returns = returns[returns < 0]
            sortino = (returns.mean() / sortino_returns.std()) * (252 ** 0.5) if len(sortino_returns) > 0 and sortino_returns.std() > 0 else 0
        else:
            sharpe = 0
            sortino = 0

        trade_stats = self.trade_log.get_stats()

        return {
            "initial_capital": self.initial_capital,
            "final_equity": self.account.equity,
            "total_return_pct": total_return,
            "max_drawdown_pct": max_drawdown,
            "sharpe_ratio": sharpe,
            "sortino_ratio": sortino,
            **trade_stats,
        }
