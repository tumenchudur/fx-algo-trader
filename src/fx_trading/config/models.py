"""
Configuration models using Pydantic.

All trading parameters, risk limits, and system settings are defined here.
"""

from datetime import datetime
from pathlib import Path
from typing import Any, Literal, Optional

import yaml
from pydantic import BaseModel, Field, field_validator, model_validator


class CostConfig(BaseModel):
    """
    Cost model configuration for realistic backtesting.

    Includes spread, commission, and slippage modeling.
    """

    # Commission settings
    commission_per_lot: float = Field(default=7.0, ge=0, description="Commission per standard lot (100k units)")
    commission_type: Literal["per_lot", "per_trade", "percentage"] = "per_lot"
    commission_percentage: float = Field(default=0.0, ge=0, description="Commission as % of trade value")

    # Slippage settings
    slippage_model: Literal["fixed_pips", "percentage", "volatility_based"] = "fixed_pips"
    slippage_pips: float = Field(default=0.5, ge=0, description="Fixed slippage in pips")
    slippage_percentage: float = Field(default=0.0001, ge=0, description="Slippage as % of price")
    slippage_volatility_multiplier: float = Field(default=0.1, ge=0, description="Multiplier for volatility-based slippage")

    # Spread settings
    use_provided_spread: bool = True  # Use bid/ask from data
    default_spread_pips: float = Field(default=1.5, ge=0, description="Default spread if not in data")
    spread_multiplier: float = Field(default=1.0, ge=1.0, description="Multiply spread in volatile conditions")


class RiskConfig(BaseModel):
    """
    Risk management configuration.

    CRITICAL: These are safety guardrails. Do not disable in production.
    """

    # Per-trade risk limits
    max_risk_per_trade_pct: float = Field(default=1.0, gt=0, le=5.0, description="Max % of equity to risk per trade")
    max_position_size_lots: float = Field(default=10.0, gt=0, description="Max lot size per position")
    min_position_size_lots: float = Field(default=0.01, gt=0, description="Min lot size (micro lots)")

    # Portfolio risk limits (high limits for leveraged trading)
    max_open_positions: int = Field(default=5, ge=1, description="Max concurrent open positions")
    max_exposure_per_currency_pct: float = Field(default=20.0, gt=0, description="Max % exposure to single currency")
    max_total_exposure_pct: float = Field(default=100.0, gt=0, description="Max total exposure as % of equity")
    max_leverage: float = Field(default=10.0, ge=1.0, le=100.0, description="Max allowed leverage")

    # Kill switches (higher limits allowed for demo/small accounts)
    daily_loss_limit_pct: float = Field(default=3.0, gt=0, le=25.0, description="Stop trading if daily loss exceeds %")
    max_drawdown_pct: float = Field(default=10.0, gt=0, le=50.0, description="Kill switch: max drawdown %")
    close_positions_on_kill: bool = Field(default=True, description="Close all positions when kill switch triggers")

    # Market condition filters
    max_spread_pips: float = Field(default=5.0, gt=0, description="Block trade if spread > threshold")
    stale_price_seconds: float = Field(default=60.0, gt=0, description="Block trade if price older than seconds")

    # Volatility filter (optional)
    enable_volatility_filter: bool = False
    max_atr_multiple: float = Field(default=3.0, gt=0, description="Block if current ATR > X * average ATR")

    # Anti-pattern protections
    no_martingale: bool = Field(default=True, description="Block doubling down after loss")
    no_revenge_trading: bool = Field(default=True, description="Block immediate re-entry after loss")
    revenge_cooldown_bars: int = Field(default=5, ge=0, description="Bars to wait after a loss")


class StrategyConfig(BaseModel):
    """Strategy configuration."""

    name: str = Field(description="Strategy identifier")
    strategy_type: Literal["volatility_breakout", "mean_reversion", "custom"] = "volatility_breakout"

    # Common parameters
    enabled: bool = True
    symbols: list[str] = Field(default_factory=lambda: ["EURUSD"])
    timeframe: str = Field(default="M5", description="Candle timeframe")

    # Entry/exit parameters
    use_stop_loss: bool = True
    use_take_profit: bool = True
    risk_reward_ratio: float = Field(default=2.0, gt=0, description="TP distance / SL distance")
    time_exit_bars: Optional[int] = Field(default=None, ge=1, description="Exit after N bars if set")

    # Strategy-specific parameters (flexible dict)
    params: dict[str, Any] = Field(default_factory=dict)

    @field_validator("symbols", mode="before")
    @classmethod
    def ensure_list(cls, v: Any) -> list[str]:
        """Ensure symbols is a list."""
        if isinstance(v, str):
            return [v]
        return v


class DataConfig(BaseModel):
    """Data ingestion and storage configuration."""

    input_path: Path = Field(description="Path to input data file or directory")
    output_path: Path = Field(default=Path("data/processed"), description="Path for processed data")
    symbol: str = Field(description="Symbol to ingest")
    timeframe: str = Field(default="M5", description="Candle timeframe")

    # Data quality settings
    timezone: str = Field(default="UTC", description="Canonical timezone")
    fill_missing: bool = Field(default=False, description="Fill missing candles")
    remove_duplicates: bool = Field(default=True, description="Remove duplicate timestamps")
    remove_outliers: bool = Field(default=True, description="Remove price outliers")
    outlier_std_threshold: float = Field(default=5.0, gt=0, description="Outlier threshold in std devs")

    # Column mapping (for custom CSV formats)
    column_mapping: dict[str, str] = Field(default_factory=dict)


class BacktestConfig(BaseModel):
    """Backtest run configuration."""

    # Run identification
    run_id: Optional[str] = Field(default=None, description="Unique run identifier")
    output_dir: Path = Field(default=Path("runs"), description="Output directory for results")

    # Data settings
    data_path: Path = Field(description="Path to processed data")
    symbols: list[str] = Field(default_factory=lambda: ["EURUSD"])
    start_date: Optional[datetime] = None
    end_date: Optional[datetime] = None

    # Account settings
    initial_capital: float = Field(default=10000.0, gt=0)
    base_currency: str = Field(default="USD")

    # Strategy
    strategy: StrategyConfig

    # Cost and risk
    costs: CostConfig = Field(default_factory=CostConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    # Execution settings
    random_seed: Optional[int] = Field(default=42, description="Seed for reproducibility")

    @field_validator("symbols", mode="before")
    @classmethod
    def ensure_symbols_list(cls, v: Any) -> list[str]:
        """Ensure symbols is a list."""
        if isinstance(v, str):
            return [v]
        return v


class WalkForwardWindow(BaseModel):
    """Single walk-forward window."""

    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime


class WalkForwardConfig(BaseModel):
    """Walk-forward analysis configuration."""

    # Base backtest config
    base_config: BacktestConfig

    # Walk-forward settings
    num_windows: int = Field(default=5, ge=2, description="Number of walk-forward windows")
    train_pct: float = Field(default=0.7, gt=0, lt=1, description="Training set percentage")
    gap_bars: int = Field(default=0, ge=0, description="Gap between train and test to prevent leakage")
    anchored: bool = Field(default=False, description="Anchored (growing train) vs rolling")

    # Optional manual windows
    windows: Optional[list[WalkForwardWindow]] = None

    @model_validator(mode="after")
    def validate_windows(self) -> "WalkForwardConfig":
        """Validate window configuration."""
        if self.windows is not None and len(self.windows) < 2:
            raise ValueError("At least 2 walk-forward windows required")
        return self


class PaperTradingConfig(BaseModel):
    """Paper trading configuration."""

    # SAFETY: This must be paper trading only
    mode: Literal["paper"] = Field(default="paper", description="Trading mode (paper only)")

    # Data source
    data_path: Path = Field(description="Path to data for paper trading")
    symbols: list[str] = Field(default_factory=lambda: ["EURUSD"])

    # Account
    initial_capital: float = Field(default=10000.0, gt=0)
    base_currency: str = Field(default="USD")

    # Strategy
    strategy: StrategyConfig

    # Cost and risk
    costs: CostConfig = Field(default_factory=CostConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    # Execution settings
    poll_interval_seconds: float = Field(default=1.0, gt=0, description="Loop interval")
    max_iterations: Optional[int] = Field(default=None, description="Max iterations (None=unlimited)")

    # Output
    output_dir: Path = Field(default=Path("runs/paper"))
    log_level: str = Field(default="INFO")

    @field_validator("mode")
    @classmethod
    def enforce_paper_mode(cls, v: str) -> str:
        """Enforce paper trading mode."""
        if v != "paper":
            raise ValueError("Only paper trading mode is allowed. Live trading requires explicit separate configuration.")
        return v


class MT5Config(BaseModel):
    """
    MetaTrader 5 connection configuration.

    Uses the official MetaTrader5 Python package for direct connection.
    Requires MT5 terminal running on the same Windows machine.

    See: https://www.mql5.com/en/docs/python_metatrader5
    """

    # Connection settings (all optional - uses terminal defaults if not specified)
    path: Optional[str] = Field(default=None, description="Path to MT5 terminal (auto-detect if None)")
    login: Optional[int] = Field(default=None, description="Account login number (use terminal's default if None)")
    password: Optional[str] = Field(default=None, description="Account password (use saved if None)")
    server: Optional[str] = Field(default=None, description="Broker server name")
    timeout_ms: int = Field(default=60000, ge=1000, description="Connection timeout in milliseconds")
    portable: bool = Field(default=False, description="Use portable mode")

    # Trading settings
    symbol_suffix: str = Field(default="", description="Suffix for symbols (e.g., 'm' for micro accounts)")
    magic_number: int = Field(default=123456, description="Magic number for identifying our trades")


class NewsFilterConfig(BaseModel):
    """
    News filter configuration for blocking trades around economic events.

    Prevents trading during high-impact news to avoid volatility spikes.

    Get a free Finnhub API key at: https://finnhub.io/
    """

    enabled: bool = Field(default=True, description="Enable news filter")
    minutes_before: int = Field(default=30, ge=0, description="Stop trading X min before event")
    minutes_after: int = Field(default=15, ge=0, description="Resume trading X min after event")
    min_impact: Literal["low", "medium", "high"] = Field(default="high", description="Minimum impact level to filter")
    currencies: Optional[list[str]] = Field(default=None, description="Currencies to monitor (None=all in traded pairs)")
    block_modifications: bool = Field(default=False, description="Also block SL/TP modifications")

    # API key for economic calendar (get free at https://finnhub.io/)
    finnhub_api_key: Optional[str] = Field(default=None, description="Finnhub API key for reliable calendar data")


class TelegramConfig(BaseModel):
    """
    Telegram notification configuration.

    Setup:
    1. Create a bot with @BotFather on Telegram - send /newbot
    2. Copy the bot token you receive
    3. Get your chat ID by messaging @userinfobot on Telegram
    4. Add both to your config
    """

    enabled: bool = Field(default=False, description="Enable Telegram notifications")
    bot_token: Optional[str] = Field(default=None, description="Bot token from @BotFather")
    chat_id: Optional[str] = Field(default=None, description="Your Telegram chat ID")

    # Notification settings
    notify_on_trade: bool = Field(default=True, description="Alert when trade opens")
    notify_on_close: bool = Field(default=True, description="Alert when trade closes")
    notify_on_error: bool = Field(default=True, description="Alert on errors")
    notify_on_kill_switch: bool = Field(default=True, description="Alert when kill switch triggers")
    notify_daily_summary: bool = Field(default=False, description="Send daily P/L summary")


class LiveTradingConfig(BaseModel):
    """
    Live/demo trading configuration.

    IMPORTANT: Start with demo account only!
    """

    # Run identification
    run_id: Optional[str] = Field(default=None, description="Unique run identifier")
    output_dir: Path = Field(default=Path("runs/live"), description="Output directory")

    # MT5 connection
    mt5: MT5Config = Field(default_factory=MT5Config)

    # Trading settings
    symbols: list[str] = Field(default_factory=lambda: ["EURUSD"])
    poll_interval_seconds: float = Field(default=5.0, gt=0, description="Main loop interval")

    # News filter
    news_filter: NewsFilterConfig = Field(default_factory=NewsFilterConfig)

    # Telegram notifications
    telegram: TelegramConfig = Field(default_factory=TelegramConfig)

    # Account
    initial_capital: Optional[float] = Field(default=None, description="Override for tracking (uses MT5 balance if None)")
    base_currency: str = Field(default="USD")

    # Strategy
    strategy: StrategyConfig

    # Cost and risk
    costs: CostConfig = Field(default_factory=CostConfig)
    risk: RiskConfig = Field(default_factory=RiskConfig)

    # Safety settings
    max_runtime_hours: Optional[float] = Field(default=None, description="Auto-stop after hours")
    close_positions_on_exit: bool = Field(default=False, description="Close all on shutdown")
    heartbeat_interval_seconds: float = Field(default=60.0, gt=0, description="Heartbeat log interval")

    # Logging
    log_level: str = Field(default="INFO")

    @field_validator("symbols", mode="before")
    @classmethod
    def ensure_symbols_list(cls, v: Any) -> list[str]:
        """Ensure symbols is a list."""
        if isinstance(v, str):
            return [v]
        return v


def load_config(path: Path, config_type: type[BaseModel]) -> BaseModel:
    """
    Load configuration from YAML file.

    Args:
        path: Path to YAML config file
        config_type: Pydantic model class to parse into

    Returns:
        Parsed configuration object
    """
    with open(path) as f:
        data = yaml.safe_load(f)

    return config_type.model_validate(data)
