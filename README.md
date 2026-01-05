# FX Trading System

A production-grade, quant-style trading system scaffold for FX (forex) trading with backtesting, risk management, and paper trading capabilities.

```
⚠️ DISCLAIMER: This is an EDUCATIONAL trading system.
   Trading forex involves significant risk of loss.
   Past performance does not guarantee future results.
   Use PAPER TRADING only. NO GUARANTEES OF PROFITABILITY.
```

## System Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FX Trading System                            │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐         │
│  │   Data       │───▶│   Strategy   │───▶│    Risk      │         │
│  │   Layer      │    │   Engine     │    │   Engine     │         │
│  └──────────────┘    └──────────────┘    └──────────────┘         │
│         │                                        │                  │
│         ▼                                        ▼                  │
│  ┌──────────────┐                        ┌──────────────┐         │
│  │   Backtest   │◀───────────────────────│  Execution   │         │
│  │   Engine     │                        │    Layer     │         │
│  └──────────────┘                        └──────────────┘         │
│         │                                        │                  │
│         ▼                                        ▼                  │
│  ┌──────────────┐                        ┌──────────────┐         │
│  │  Portfolio   │                        │   Paper      │         │
│  │  Accounting  │                        │   Broker     │         │
│  └──────────────┘                        └──────────────┘         │
│         │                                                          │
│         ▼                                                          │
│  ┌──────────────────────────────────────────────────────┐         │
│  │              Monitoring & Reporting                   │         │
│  └──────────────────────────────────────────────────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

## Features

### Core Components

- **Data Layer**: OHLCV ingestion, validation, bid/ask support, Parquet storage
- **Strategy Engine**: Configurable rule-based strategies with no lookahead bias
- **Risk Engine**: Pre-trade checks, kill switches, position sizing
- **Backtesting**: Realistic simulation with spread, slippage, and commission
- **Walk-Forward**: Out-of-sample testing with rolling/anchored windows
- **Execution**: Abstract broker interface with paper trading implementation

### Safety Features

-  Daily loss limit kill switch
-  Max drawdown kill switch
-  Spread filter (blocks wide spreads)
-  Stale price rejection
-  Anti-martingale protection
-  Anti-revenge trading cooldown
-  Position size limits
-  Leverage limits
-  Default paper trading only

## Installation

### Prerequisites

- Python 3.11+
- pip or uv package manager

### Install from source

```bash
# Clone the repository
cd quant-trading

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install package
pip install -e .

# Install dev dependencies
pip install -e ".[dev]"
```

## Quickstart

### 1. Generate Sample Data

```bash
# Generate synthetic EURUSD data
fx-trading generate-sample --symbol EURUSD --bars 5000 --output data/sample
```

### 2. Run a Backtest

```bash
# Run backtest with example config
fx-trading backtest configs/backtest_example.yaml

# View results
open runs/<run_id>/report.html
```

### 3. Run Walk-Forward Analysis

```bash
fx-trading walkforward configs/walkforward_example.yaml
```

### 4. Paper Trading Simulation

```bash
fx-trading paper-trade configs/paper_example.yaml
```

## CLI Commands

| Command | Description |
|---------|-------------|
| `fx-trading ingest-data` | Ingest OHLCV data from CSV |
| `fx-trading generate-sample` | Generate synthetic test data |
| `fx-trading backtest` | Run a backtest |
| `fx-trading walkforward` | Run walk-forward analysis |
| `fx-trading paper-trade` | Run paper trading simulation |
| `fx-trading report` | Generate report for a run |
| `fx-trading validate-config` | Validate a config file |
| `fx-trading version` | Show version |

## Configuration

All settings are defined in YAML files. See `configs/` for examples.

### Key Configuration Sections

```yaml
# Strategy settings
strategy:
  name: my_strategy
  strategy_type: volatility_breakout
  params:
    lookback: 20
    atr_period: 14

# Cost model
costs:
  commission_per_lot: 7.0
  slippage_pips: 0.5

# Risk limits (CRITICAL)
risk:
  max_risk_per_trade_pct: 1.0    # Max 1% equity per trade
  daily_loss_limit_pct: 3.0      # Kill switch at 3% daily loss
  max_drawdown_pct: 10.0         # Kill switch at 10% drawdown
  max_spread_pips: 3.0           # Skip if spread too wide
```

## Cost Modeling

The system models three types of execution costs:

### 1. Spread

Bid/ask spread is applied to all trades:
- **Long entry**: Execute at ask (higher price)
- **Long exit**: Execute at bid (lower price)
- **Short entry**: Execute at bid
- **Short exit**: Execute at ask

### 2. Slippage

Additional price impact modeled as:
- Fixed pips (default)
- Percentage of price
- Volatility-based (ATR multiple)

Slippage always works against the trader.

### 3. Commission

Per-lot or percentage-based commission applied to each trade.

**Example**: For a 1-lot EURUSD trade with 2-pip spread, 0.5-pip slippage, and $7 commission:
- Spread cost: ~$20
- Slippage cost: ~$5
- Commission: $7
- Total round-trip cost: ~$32

## Risk Controls

### Pre-Trade Checks

| Check | Description |
|-------|-------------|
| Spread filter | Block if spread > threshold |
| Stale price | Block if price data too old |
| Max positions | Block if at position limit |
| Existing position | Block duplicate symbol |
| Exposure limit | Reduce size if exposure high |
| Leverage limit | Reduce size if leverage high |

### Kill Switches

| Trigger | Action |
|---------|--------|
| Daily loss > limit | Stop trading, optionally close all |
| Drawdown > limit | Stop trading, optionally close all |

### Position Sizing

Risk-based sizing using stop loss distance:
```
Size = (Equity × Risk%) / (SL_distance × Pip_value_per_lot)
```

Constrained by min/max lot sizes.

## Project Structure

```
quant-trading/
├── src/fx_trading/
│   ├── backtesting/     # Backtest and walk-forward engines
│   ├── config/          # Pydantic config models
│   ├── costs/           # Cost models (spread, slippage, commission)
│   ├── data/            # Data ingestion and validation
│   ├── execution/       # Broker interfaces
│   ├── monitoring/      # Logging and reports
│   ├── portfolio/       # Accounting and position sizing
│   ├── risk/            # Risk engine
│   ├── strategies/      # Strategy implementations
│   ├── types/           # Core data types
│   └── cli.py           # CLI interface
├── tests/               # Pytest tests
├── configs/             # Example configurations
├── data/                # Data storage
└── runs/                # Backtest results
```

## Adding a New Strategy

1. Create a new file in `src/fx_trading/strategies/`:

```python
from fx_trading.strategies.base import Strategy
from fx_trading.types.models import Signal, Side

class MyStrategy(Strategy):
    def generate_signals(self, data, current_index):
        # Only use data up to current_index (no lookahead!)
        historical = data.iloc[:current_index + 1]

        # Your logic here...

        if should_go_long:
            return [Signal(
                timestamp=data.index[current_index],
                symbol=self.symbols[0],
                side=Side.LONG,
                stop_loss=calculate_sl(),
                take_profit=calculate_tp(),
            )]
        return []
```

2. Register in `strategies/__init__.py`

3. Use `strategy_type: "my_strategy"` in config

## Adding a Real Broker

The `LiveBrokerStub` shows the interface. To implement:

1. Create a new class implementing the `Broker` interface
2. Implement all abstract methods (`connect`, `place_order`, etc.)
3. Add proper authentication and error handling
4. **TEST THOROUGHLY IN PAPER MODE FIRST**

See `src/fx_trading/execution/live_broker_stub.py` for the TODO list.

## Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=fx_trading

# Run specific test file
pytest tests/test_risk_engine.py

# Run in verbose mode
pytest -v
```

## Development

```bash
# Format code
black src tests

# Lint
ruff check src tests

# Type check
mypy src
```

## TODO: Production Considerations

Before using this for real trading (NOT RECOMMENDED):

- [ ] Implement real broker adapter (OANDA, Interactive Brokers, etc.)
- [ ] Add proper error handling and retry logic
- [ ] Implement order reconciliation
- [ ] Add position synchronization
- [ ] Set up monitoring and alerting
- [ ] Implement proper logging infrastructure
- [ ] Add circuit breakers for API failures
- [ ] Test extensively in paper mode
- [ ] Get professional financial advice
- [ ] Understand all risks involved

## License

MIT License - See LICENSE file.

## Contributing

This is an educational project. Contributions welcome for:
- Bug fixes
- Documentation improvements
- Additional test coverage
- New strategy implementations (educational)

## Acknowledgments

Built as an educational scaffold demonstrating quant trading system design patterns.

---

**Remember: This is for EDUCATIONAL purposes only. Paper trading is the only supported mode. Trading forex involves substantial risk of loss.**
