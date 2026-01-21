"""
Position Management: Trailing Stop and Partial Take Profit.

Provides intelligent exit management for open positions:
- TrailingStopManager: Dynamically adjusts stop loss as price moves in profit
- PartialTakeProfitManager: Closes partial position at predefined targets
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional
from uuid import UUID

from loguru import logger

from fx_trading.types.models import Position, Side


@dataclass
class TrailingStopConfig:
    """Configuration for trailing stop behavior."""

    enabled: bool = False
    method: str = "atr_based"  # "atr_based" or "fixed_pips"
    atr_multiplier: float = 2.0  # Trail at N * ATR behind price
    fixed_pips: float = 30.0  # Fixed pip distance for trail
    activation_profit_atr: float = 1.0  # Activate after N * ATR profit
    step_pips: float = 5.0  # Minimum SL improvement in pips


@dataclass
class PartialTPConfig:
    """Configuration for partial take profit."""

    enabled: bool = False
    first_target_r: float = 1.0  # First partial at 1R profit
    first_close_pct: float = 50.0  # Close 50% at first target
    move_sl_to_breakeven: bool = True  # Move SL to entry after first partial
    second_target_r: Optional[float] = None  # Optional second partial
    second_close_pct: float = 25.0  # Close 25% at second target


@dataclass
class PositionState:
    """Internal state tracking for position management."""

    position_id: UUID
    peak_price: float = 0.0  # Best price achieved (for trailing)
    entry_price: float = 0.0
    initial_sl: Optional[float] = None
    initial_risk: float = 0.0  # Distance from entry to initial SL
    partials_taken: int = 0
    remaining_size: float = 0.0
    sl_moved_to_breakeven: bool = False


@dataclass
class PartialExitAction:
    """Action to take for partial exit."""

    close_size: float
    new_sl: Optional[float] = None
    reason: str = ""


class TrailingStopManager:
    """
    Manages trailing stops for open positions.

    The trailing stop locks in profits by moving the stop loss
    as price moves in the position's favor.

    Supports two trailing methods:
    - ATR-based: Trail at N * ATR behind the peak price
    - Fixed pips: Trail at a fixed pip distance behind peak

    Only activates after position achieves minimum profit threshold.
    """

    def __init__(self, config: TrailingStopConfig):
        """
        Initialize trailing stop manager.

        Args:
            config: Trailing stop configuration
        """
        self.config = config
        self._state: dict[UUID, PositionState] = {}

    def register_position(self, position: Position) -> None:
        """
        Register a new position for trailing.

        Args:
            position: Position to track
        """
        state = PositionState(
            position_id=position.id,
            peak_price=position.entry_price,
            entry_price=position.entry_price,
            initial_sl=position.stop_loss,
            initial_risk=abs(position.entry_price - position.stop_loss) if position.stop_loss else 0,
            remaining_size=position.size,
        )
        self._state[position.id] = state
        logger.debug(f"Registered position {position.id} for trailing stop")

    def unregister_position(self, position_id: UUID) -> None:
        """Remove position from tracking."""
        if position_id in self._state:
            del self._state[position_id]

    def calculate_new_stop(
        self,
        position: Position,
        current_price: float,
        atr: Optional[float] = None,
    ) -> Optional[float]:
        """
        Calculate new trailing stop level if improvement is possible.

        Args:
            position: Current position
            current_price: Current market price
            atr: Current ATR value (required for atr_based method)

        Returns:
            New stop loss price if trail should be moved, None otherwise
        """
        if not self.config.enabled:
            return None

        # Get or create state
        if position.id not in self._state:
            self.register_position(position)

        state = self._state[position.id]

        # Determine trailing distance
        if self.config.method == "atr_based":
            if atr is None:
                logger.warning("ATR required for atr_based trailing but not provided")
                return None
            trail_distance = atr * self.config.atr_multiplier
            activation_distance = atr * self.config.activation_profit_atr
        else:  # fixed_pips
            # Convert pips to price (assume 4 decimal pairs, adjust for JPY/Gold)
            pip_value = 0.0001
            if "JPY" in position.symbol:
                pip_value = 0.01
            elif "XAU" in position.symbol or "GOLD" in position.symbol.upper():
                pip_value = 0.01
            trail_distance = self.config.fixed_pips * pip_value
            activation_distance = trail_distance

        # Update peak price
        if position.side == Side.LONG:
            if current_price > state.peak_price:
                state.peak_price = current_price

            # Check if profit threshold met
            profit = current_price - state.entry_price
            if profit < activation_distance:
                return None  # Not enough profit to activate trailing

            # Calculate new stop
            new_sl = state.peak_price - trail_distance

        else:  # SHORT
            if current_price < state.peak_price or state.peak_price == state.entry_price:
                state.peak_price = min(state.peak_price, current_price) if state.peak_price != state.entry_price else current_price

            # Check if profit threshold met
            profit = state.entry_price - current_price
            if profit < activation_distance:
                return None  # Not enough profit to activate trailing

            # Calculate new stop
            new_sl = state.peak_price + trail_distance

        # Only move stop if it's an improvement
        current_sl = position.stop_loss
        if current_sl is None:
            return new_sl

        # Check minimum step (in pips)
        pip_value = 0.0001
        if "JPY" in position.symbol:
            pip_value = 0.01
        elif "XAU" in position.symbol or "GOLD" in position.symbol.upper():
            pip_value = 0.01
        min_step = self.config.step_pips * pip_value

        if position.side == Side.LONG:
            improvement = new_sl - current_sl
            if improvement >= min_step:
                logger.info(
                    f"Trailing stop update for {position.symbol}: "
                    f"{current_sl:.5f} -> {new_sl:.5f} (+{improvement/pip_value:.1f} pips)"
                )
                return new_sl
        else:  # SHORT
            improvement = current_sl - new_sl
            if improvement >= min_step:
                logger.info(
                    f"Trailing stop update for {position.symbol}: "
                    f"{current_sl:.5f} -> {new_sl:.5f} (+{improvement/pip_value:.1f} pips)"
                )
                return new_sl

        return None


class PartialTakeProfitManager:
    """
    Manages partial take profit exits.

    Closes a portion of the position at predefined profit targets,
    allowing the remaining position to run for larger gains while
    locking in some profit.

    Features:
    - First partial at 1R (risk-reward 1:1)
    - Optional second partial at 2R
    - Move stop to breakeven after first partial
    """

    def __init__(self, config: PartialTPConfig):
        """
        Initialize partial TP manager.

        Args:
            config: Partial take profit configuration
        """
        self.config = config
        self._state: dict[UUID, PositionState] = {}

    def register_position(self, position: Position) -> None:
        """
        Register a new position for partial TP tracking.

        Args:
            position: Position to track
        """
        initial_risk = 0.0
        if position.stop_loss:
            initial_risk = abs(position.entry_price - position.stop_loss)

        state = PositionState(
            position_id=position.id,
            entry_price=position.entry_price,
            initial_sl=position.stop_loss,
            initial_risk=initial_risk,
            remaining_size=position.size,
        )
        self._state[position.id] = state
        logger.debug(f"Registered position {position.id} for partial TP")

    def unregister_position(self, position_id: UUID) -> None:
        """Remove position from tracking."""
        if position_id in self._state:
            del self._state[position_id]

    def check_partial_exit(
        self,
        position: Position,
        current_price: float,
    ) -> Optional[PartialExitAction]:
        """
        Check if partial take profit should be executed.

        Args:
            position: Current position
            current_price: Current market price

        Returns:
            PartialExitAction if partial should be taken, None otherwise
        """
        if not self.config.enabled:
            return None

        # Get or create state
        if position.id not in self._state:
            self.register_position(position)

        state = self._state[position.id]

        if state.initial_risk <= 0:
            return None  # Can't calculate R without initial risk

        # Calculate current profit in R multiples
        if position.side == Side.LONG:
            profit = current_price - state.entry_price
        else:
            profit = state.entry_price - current_price

        r_multiple = profit / state.initial_risk

        # Check first partial
        if state.partials_taken == 0 and r_multiple >= self.config.first_target_r:
            close_size = state.remaining_size * (self.config.first_close_pct / 100)

            # Determine new SL
            new_sl = None
            if self.config.move_sl_to_breakeven:
                new_sl = state.entry_price
                state.sl_moved_to_breakeven = True

            state.partials_taken = 1
            state.remaining_size -= close_size

            logger.info(
                f"Partial TP triggered for {position.symbol}: "
                f"closing {close_size:.2f} lots @ {r_multiple:.1f}R, "
                f"remaining {state.remaining_size:.2f} lots"
            )

            return PartialExitAction(
                close_size=close_size,
                new_sl=new_sl,
                reason=f"Partial TP at {self.config.first_target_r}R",
            )

        # Check second partial if configured
        if (
            self.config.second_target_r
            and state.partials_taken == 1
            and r_multiple >= self.config.second_target_r
        ):
            close_size = state.remaining_size * (self.config.second_close_pct / 100)

            state.partials_taken = 2
            state.remaining_size -= close_size

            logger.info(
                f"Second partial TP for {position.symbol}: "
                f"closing {close_size:.2f} lots @ {r_multiple:.1f}R, "
                f"remaining {state.remaining_size:.2f} lots"
            )

            return PartialExitAction(
                close_size=close_size,
                new_sl=None,
                reason=f"Partial TP at {self.config.second_target_r}R",
            )

        return None

    def get_state(self, position_id: UUID) -> Optional[PositionState]:
        """Get current state for a position."""
        return self._state.get(position_id)


def parse_trailing_stop_config(params: dict) -> TrailingStopConfig:
    """
    Parse trailing stop config from strategy params.

    Args:
        params: Strategy params dict

    Returns:
        TrailingStopConfig instance
    """
    ts_config = params.get("trailing_stop", {})
    if not ts_config:
        return TrailingStopConfig(enabled=False)

    return TrailingStopConfig(
        enabled=ts_config.get("enabled", False),
        method=ts_config.get("method", "atr_based"),
        atr_multiplier=ts_config.get("atr_multiplier", 2.0),
        fixed_pips=ts_config.get("fixed_pips", 30.0),
        activation_profit_atr=ts_config.get("activation_profit_atr", 1.0),
        step_pips=ts_config.get("step_pips", 5.0),
    )


def parse_partial_tp_config(params: dict) -> PartialTPConfig:
    """
    Parse partial TP config from strategy params.

    Args:
        params: Strategy params dict

    Returns:
        PartialTPConfig instance
    """
    ptp_config = params.get("partial_tp", {})
    if not ptp_config:
        return PartialTPConfig(enabled=False)

    return PartialTPConfig(
        enabled=ptp_config.get("enabled", False),
        first_target_r=ptp_config.get("first_target_r", 1.0),
        first_close_pct=ptp_config.get("first_close_pct", 50.0),
        move_sl_to_breakeven=ptp_config.get("move_sl_to_breakeven", True),
        second_target_r=ptp_config.get("second_target_r"),
        second_close_pct=ptp_config.get("second_close_pct", 25.0),
    )
