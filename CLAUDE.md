# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Read this first

`RESEARCH.md` at repo root is the **definitive record of what has been tested and what works**. Before suggesting that the user tune `volatility_breakout`, add a new indicator filter, or otherwise extend the bundled strategy, read it. The bundled strategy was validated empirically as a negative-expectancy generator on M5/H1 majors at realistic costs; the only validated edge is a vol-targeted FX carry basket on D1 (Sharpe 0.26 NET, 0.37 OOS). Do not propose strategy work that contradicts those findings without a concrete reason the prior work was wrong.

The 9 research scripts under `scripts/edge_*.py`, `scripts/carry_*.py`, `scripts/download_*.py`, and `scripts/parse_bis_rates.py` are reusable. Any new signal idea should be tested through one of those (or a new sibling) using the same methodology — non-overlapping trades, IS/OOS split, realistic costs, Welch verification of survivors — before wiring it into `src/fx_trading/strategies/`.

## Common commands

Install (editable, with dev tooling):
```bash
pip install -e ".[dev]"
```

Test, lint, type-check, format (also exposed via `make test|lint|typecheck|format|check`):
```bash
pytest                                # full suite (pyproject sets -v --tb=short)
pytest tests/test_risk_engine.py      # single file
pytest tests/test_risk_engine.py::TestRiskEngine::test_kill_switch  # single test
pytest --cov=fx_trading                # coverage
ruff check src tests                   # lint (line length 100, ignored E501)
black src tests                        # format (line length 100)
mypy src                               # strict: disallow_untyped_defs, warn_return_any
```

The CLI is installed as `fx-trading` (entry point `fx_trading.cli:app`, Typer). Common subcommands:
```bash
fx-trading generate-sample --symbol EURUSD --bars 5000 --output data/sample
fx-trading backtest configs/backtest_optimized.yaml
fx-trading walkforward configs/walkforward_example.yaml
fx-trading paper-trade configs/paper_example.yaml
fx-trading mt5-status                  # check MT5 broker connectivity
fx-trading live configs/<live>.yaml    # gated by safety checks (see trading/safety.py)
fx-trading validate-config <path>      # parse a YAML against the Pydantic models
fx-trading report <run_id>             # regenerate HTML report for a run
```

`make demo`, `make backtest`, `make walkforward` chain `generate-sample` + the relevant CLI step. Backtest output lands in `runs/<run_id>/` (trades CSV, equity curve, `report.html`).

## Architecture

The system is a single Python package (`src/fx_trading/`) wired together by Pydantic config models. The backtest engine, paper broker, and live runner all consume the **same** strategy / risk / portfolio components — the only difference is where price data and fills come from. Keep that symmetry in mind when changing any shared component.

### Data flow per bar (Backtester / live loop)

1. `Backtester.run()` (`backtesting/engine.py`) iterates bars. For each bar it builds a `PriceData` (bid/ask/mid), advances `PaperBroker.set_prices` + `set_bar_index`, and runs trailing-stop / time-exit / SL-TP fills before generating new signals.
2. `Strategy.generate_signals(data, current_index)` produces `Signal` objects. **Strategies must only read `data.iloc[: current_index + 1]`** — accessing future bars is lookahead bias and silently corrupts results. The base class enforces nothing; reviewers must.
3. `RiskEngine.evaluate_signal()` (`risk/engine.py`) runs pre-trade checks in order: stale price, spread, max positions, daily-loss / drawdown kill switches, exposure & leverage caps, anti-martingale / revenge-trading cooldown, and finally **risk-adjusted sizing** that scales the lot size from `max_risk_per_trade_pct`, the SL distance, and the symbol's pip value. The engine can also activate a kill switch that flags `should_close_all_positions()`.
4. Approved orders go through `Broker.place_order` (`PaperBroker` in tests/backtests, `MT5Broker` in live). The paper broker applies spread + slippage + commission via `costs/models.FillCalculator`.
5. `PortfolioManager` (`portfolio/accounting.py`) records fills, marks positions to market, and feeds equity back into `RiskEngine.record_trade_result` so the kill switches and cooldown timers stay coherent.
6. After the loop, `Backtester._compile_results` writes a `BacktestResult` (trades_df, equity_curve, cost breakdown, sharpe/sortino, etc.) and `monitoring/` renders the HTML report.

### Configuration is the source of truth

Every behaviour switch lives in `src/fx_trading/config/models.py` (Pydantic v2). YAML files in `configs/` are loaded directly into these models — when adding a feature, add a typed field there first; do not read raw dict keys in strategy / risk code. Notable shape:

- `StrategyConfig.params` is a free-form `dict[str, Any]`; strategy-specific knobs (lookback, ATR period, filter toggles, `allowed_sides`, `min_atr_percentile`, etc.) live there. `StrategyConfig.symbol_params` overrides `params` per symbol via `get_params_for_symbol(symbol)` — strategies must call `self.get_param(key, symbol=...)` rather than reading `self.params` directly when behaviour is per-symbol.
- `RiskConfig` is the only place to define new pre-trade checks or kill-switch thresholds; wire them into `RiskEngine.evaluate_signal` in the same order as existing checks so logs/decisions stay consistent.
- `CostConfig` controls spread/slippage/commission for both backtest and paper. The backtester respects `use_provided_spread` and falls back to `default_spread_pips` when bid/ask is absent.

### Adding a strategy

1. Subclass `Strategy` in `src/fx_trading/strategies/` and implement `generate_signals`. Use `self.calculate_stop_loss` / `calculate_take_profit` from the base class so SL/TP semantics stay uniform.
2. Register it both in `strategies/__init__.py` (eager import) and in the lazy import branch inside `StrategyFactory.create` (`strategies/base.py`) — the factory has a hardcoded if/elif for the built-in types.
3. Add the new name to the `Literal[...]` in `StrategyConfig.strategy_type` (`config/models.py`), otherwise YAML validation rejects it.

### Brokers

`execution/broker.py` defines the abstract `Broker` interface; `paper_broker.py` is the reference implementation. `mt5_broker.py` / `mt5_client.py` / `mt5_zmq_client.py` are MT5 adapters (live trading is opt-in and gated by `trading/safety.py` plus `LIVE_TRADING_ENABLED` in `fx_trading/__init__.py`). Any new broker must satisfy the full interface — the backtester calls `set_prices`, `set_bar_index`, `update_positions`, `check_pending_orders`, and `check_time_exits` in a specific order each bar; do not change that contract without updating both engines.

### Tests

`tests/conftest.py` provides shared fixtures (synthetic OHLCV, configured risk engine). Integration coverage in `test_integration.py` exercises a full backtest end-to-end; prefer extending it over writing one-off harness scripts. Strategy and risk tests are the canonical examples for how to construct configs in code.

## Repository conventions worth knowing

- Python 3.11+. Pydantic v2 throughout — use `model_validator` / `field_validator`, not v1 idioms.
- Logging is `loguru` (`from loguru import logger`); do not introduce stdlib `logging` handlers.
- Prices/sizes are floats; pip values and lot sizes are computed in `portfolio/position_sizing.py`. JPY pairs use a 0.01 pip definition — re-use the existing helpers instead of hardcoding 0.0001.
- `runs/` and `data/` are gitignored output sinks; don't commit generated CSV/Parquet/HTML.
- `configs/mt5_demo.yaml` was previously committed with credentials and is now untracked — never re-add it. Use `configs/mt5_demo.example.yaml` as the template.
- Default mode is paper trading. The README and `__init__.py` both gate live trading behind explicit opt-in; preserve that posture in any new entry point.
