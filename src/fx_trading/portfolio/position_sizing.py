"""
Position sizing based on risk management.

Calculates appropriate position sizes based on risk per trade.
"""

from typing import Optional

from loguru import logger

from fx_trading.config.models import RiskConfig
from fx_trading.types.models import Side, get_contract_size


class PositionSizer:
    """
    Calculate position sizes based on risk parameters.

    Uses risk-based sizing with stop loss distance.
    """

    def __init__(
        self,
        config: RiskConfig,
        account_currency: str = "USD",
    ):
        """
        Initialize position sizer.

        Args:
            config: Risk configuration
            account_currency: Account base currency
        """
        self.config = config
        self.account_currency = account_currency

    def calculate_size(
        self,
        equity: float,
        entry_price: float,
        stop_loss: Optional[float],
        side: Side,
        symbol: str = "EURUSD",
        pip_value: float = 0.0001,
    ) -> float:
        """
        Calculate position size based on risk.

        Risk-based sizing: size = (equity * risk_pct) / (SL_distance_in_pips * pip_value_per_lot)

        Args:
            equity: Current account equity
            entry_price: Expected entry price
            stop_loss: Stop loss price
            side: Trade side
            symbol: Trading symbol
            pip_value: Value of 1 pip

        Returns:
            Position size in lots (constrained by min/max)
        """
        # If no stop loss, use max position size
        if stop_loss is None:
            logger.warning("No stop loss provided, using max position size")
            return self._constrain_size(self.config.max_position_size_lots)

        # Calculate stop loss distance in pips
        if side == Side.LONG:
            sl_distance = entry_price - stop_loss
        else:
            sl_distance = stop_loss - entry_price

        if sl_distance <= 0:
            logger.warning(f"Invalid stop loss distance: {sl_distance}")
            return self.config.min_position_size_lots

        sl_pips = sl_distance / pip_value

        # Calculate risk amount in account currency
        risk_amount = equity * (self.config.max_risk_per_trade_pct / 100)

        # Standard lot = 100,000 units
        # Pip value per lot = 100,000 * pip_value = 10 for most pairs
        pip_value_per_lot = 100000 * pip_value

        # For pairs where quote currency != account currency, would need conversion
        # Simplified: assume USD account and major pairs
        if "XAU" in symbol or "GOLD" in symbol.upper():
            # Gold: 1 lot = 100 oz, pip = $0.01, so $1 per pip per lot
            pip_value_per_lot = 1.0
            # Recalculate sl_pips for gold (pip = $0.01)
            sl_pips = sl_distance / 0.01
        elif "JPY" in symbol:
            # JPY pip value different
            pip_value_per_lot = 100000 * 0.01 / entry_price  # Approximate conversion
        else:
            pip_value_per_lot = 10  # $10 per pip per standard lot for USD pairs

        # Calculate size
        if sl_pips * pip_value_per_lot <= 0:
            logger.warning("Invalid pip value calculation")
            return self.config.min_position_size_lots

        size = risk_amount / (sl_pips * pip_value_per_lot)

        logger.debug(
            f"Position sizing: equity={equity:.2f} risk={self.config.max_risk_per_trade_pct}% "
            f"SL_pips={sl_pips:.1f} -> size={size:.4f} lots"
        )

        return self._constrain_size(size)

    def calculate_fixed_risk_size(
        self,
        equity: float,
        risk_pips: float,
        symbol: str = "EURUSD",
    ) -> float:
        """
        Calculate size for a fixed pip risk.

        Args:
            equity: Account equity
            risk_pips: Stop loss in pips
            symbol: Trading symbol

        Returns:
            Position size in lots
        """
        risk_amount = equity * (self.config.max_risk_per_trade_pct / 100)

        # Pip value per lot based on symbol type
        if "XAU" in symbol or "GOLD" in symbol.upper():
            pip_value_per_lot = 1.0  # Gold: $1 per pip per lot
        elif "JPY" in symbol:
            pip_value_per_lot = 1000 / 150  # Approximate for JPY pairs
        else:
            pip_value_per_lot = 10  # Standard forex pairs

        size = risk_amount / (risk_pips * pip_value_per_lot)
        return self._constrain_size(size)

    def calculate_volatility_adjusted_size(
        self,
        equity: float,
        atr: float,
        atr_multiplier: float = 2.0,
        entry_price: float = 1.0,
        symbol: str = "EURUSD",
    ) -> float:
        """
        Calculate size based on ATR volatility.

        Uses ATR to determine stop loss distance, then calculates size.

        Args:
            equity: Account equity
            atr: Current ATR value
            atr_multiplier: ATR multiple for stop loss
            entry_price: Entry price (for pip calculation)
            symbol: Trading symbol

        Returns:
            Position size in lots
        """
        # Determine pip value based on symbol type
        if "XAU" in symbol or "GOLD" in symbol.upper():
            pip_value = 0.01  # Gold pip = $0.01
        elif "JPY" in symbol:
            pip_value = 0.01  # JPY pairs
        else:
            pip_value = 0.0001  # Standard forex

        # SL distance = ATR * multiplier
        sl_distance = atr * atr_multiplier
        sl_pips = sl_distance / pip_value

        return self.calculate_fixed_risk_size(equity, sl_pips, symbol)

    def _constrain_size(self, size: float) -> float:
        """
        Constrain size to min/max limits.

        Args:
            size: Calculated size

        Returns:
            Constrained size
        """
        # Round to standard lot increments (micro lots = 0.01)
        size = round(size, 2)

        # Apply limits
        size = max(size, self.config.min_position_size_lots)
        size = min(size, self.config.max_position_size_lots)

        return size

    def check_exposure_limit(
        self,
        proposed_size: float,
        current_exposure: float,
        equity: float,
        entry_price: float,
        symbol: str = "EURUSD",
    ) -> tuple[float, bool]:
        """
        Check if proposed size exceeds exposure limits.

        Args:
            proposed_size: Proposed position size
            current_exposure: Current total exposure
            equity: Account equity
            entry_price: Entry price
            symbol: Trading symbol (for contract size)

        Returns:
            Tuple of (adjusted_size, was_limited)
        """
        # Calculate proposed exposure using symbol-specific contract size
        contract_size = get_contract_size(symbol)
        proposed_exposure = proposed_size * contract_size * entry_price
        total_exposure = current_exposure + proposed_exposure

        # Check max total exposure
        max_exposure = equity * (self.config.max_total_exposure_pct / 100)

        if total_exposure > max_exposure:
            # Reduce size to fit limit
            available = max_exposure - current_exposure
            if available <= 0:
                return 0.0, True

            adjusted_size = available / (contract_size * entry_price)
            adjusted_size = self._constrain_size(adjusted_size)

            logger.warning(
                f"Exposure limit reached: reduced size from {proposed_size:.2f} to {adjusted_size:.2f}"
            )
            return adjusted_size, True

        return proposed_size, False

    def check_leverage_limit(
        self,
        proposed_size: float,
        current_exposure: float,
        equity: float,
        entry_price: float,
        symbol: str = "EURUSD",
    ) -> tuple[float, bool]:
        """
        Check if proposed size exceeds leverage limit.

        Args:
            proposed_size: Proposed position size
            current_exposure: Current total exposure
            equity: Account equity
            entry_price: Entry price
            symbol: Trading symbol (for contract size)

        Returns:
            Tuple of (adjusted_size, was_limited)
        """
        contract_size = get_contract_size(symbol)
        proposed_exposure = proposed_size * contract_size * entry_price
        total_exposure = current_exposure + proposed_exposure

        current_leverage = total_exposure / equity if equity > 0 else 0

        if current_leverage > self.config.max_leverage:
            # Calculate max allowed exposure
            max_exposure = equity * self.config.max_leverage
            available = max_exposure - current_exposure

            if available <= 0:
                return 0.0, True

            adjusted_size = available / (contract_size * entry_price)
            adjusted_size = self._constrain_size(adjusted_size)

            logger.warning(
                f"Leverage limit reached: reduced size from {proposed_size:.2f} to {adjusted_size:.2f}"
            )
            return adjusted_size, True

        return proposed_size, False
