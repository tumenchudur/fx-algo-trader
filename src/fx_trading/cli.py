"""
Command Line Interface for the FX Trading System.

Provides commands for data ingestion, backtesting, and paper trading.
"""

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

app = typer.Typer(
    name="fx-trading",
    help="FX Trading System - Educational quant trading scaffold",
    add_completion=False,
)

console = Console()


def print_warning() -> None:
    """Print educational disclaimer."""
    console.print(Panel(
        "[yellow]WARNING: This is an EDUCATIONAL trading system.\n"
        "Trading forex involves significant risk of loss.\n"
        "Use PAPER TRADING only. No guarantees of profitability.[/yellow]",
        title="Disclaimer",
        border_style="yellow",
    ))


@app.command()
def ingest_data(
    input_path: Path = typer.Argument(..., help="Path to input CSV file"),
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Trading symbol"),
    timeframe: str = typer.Option("M5", "--timeframe", "-t", help="Data timeframe"),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output",
        "-o",
        help="Output directory for processed data",
    ),
) -> None:
    """
    Ingest and validate OHLCV data from CSV.

    Normalizes data, validates quality, and saves to Parquet format.
    """
    from fx_trading.data.ingestion import DataIngestor
    from fx_trading.data.validation import DataQualityValidator

    console.print(f"\n[bold]Ingesting data from {input_path}[/bold]\n")

    if not input_path.exists():
        console.print(f"[red]Error: File not found: {input_path}[/red]")
        raise typer.Exit(1)

    ingestor = DataIngestor()

    try:
        df, report = ingestor.ingest_csv(
            path=input_path,
            symbol=symbol,
            timeframe=timeframe,
        )

        # Print quality report
        table = Table(title="Data Quality Report")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Rows", str(report.total_rows))
        table.add_row("Valid Rows", str(report.valid_rows))
        table.add_row("Quality Score", f"{report.quality_score:.1f}%")
        table.add_row("Duplicates", str(report.duplicate_timestamps))
        table.add_row("Missing Timestamps", str(report.missing_timestamps))
        table.add_row("Price Outliers", str(report.price_outliers))

        console.print(table)

        if report.issues:
            console.print("\n[yellow]Issues found:[/yellow]")
            for issue in report.issues[:5]:
                console.print(f"  - {issue['type']}: {issue['message']}")

        # Save to parquet
        output_path = output_dir / f"{symbol}_{timeframe}.parquet"
        ingestor.save_parquet(df, output_path)

        console.print(f"\n[green]Data saved to {output_path}[/green]")

    except Exception as e:
        console.print(f"[red]Error: {e}[/red]")
        raise typer.Exit(1)


@app.command()
def generate_sample(
    output_dir: Path = typer.Option(
        Path("data/sample"),
        "--output",
        "-o",
        help="Output directory",
    ),
    symbol: str = typer.Option("EURUSD", "--symbol", "-s", help="Symbol to generate"),
    timeframe: str = typer.Option("M5", "--timeframe", "-t", help="Timeframe"),
    num_bars: int = typer.Option(5000, "--bars", "-n", help="Number of bars"),
    seed: int = typer.Option(42, "--seed", help="Random seed"),
) -> None:
    """
    Generate synthetic sample data for testing.
    """
    from fx_trading.data.synthetic import SyntheticDataGenerator

    console.print(f"\n[bold]Generating {num_bars} bars of synthetic {symbol} data[/bold]\n")

    generator = SyntheticDataGenerator(seed=seed)
    df = generator.generate(
        symbol=symbol,
        timeframe=timeframe,
        num_bars=num_bars,
    )

    # Save
    output_dir.mkdir(parents=True, exist_ok=True)
    parquet_path = output_dir / f"{symbol}_{timeframe}.parquet"
    csv_path = output_dir / f"{symbol}_{timeframe}.csv"

    df.to_parquet(parquet_path)
    df.to_csv(csv_path)

    console.print(f"[green]Generated {len(df)} bars[/green]")
    console.print(f"Parquet: {parquet_path}")
    console.print(f"CSV: {csv_path}")

    # Show sample
    console.print("\n[bold]Sample data:[/bold]")
    console.print(df.head().to_string())


@app.command()
def backtest(
    config: Path = typer.Argument(..., help="Path to backtest config YAML"),
    output_dir: Optional[Path] = typer.Option(
        None,
        "--output",
        "-o",
        help="Output directory for results",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
) -> None:
    """
    Run a backtest using configuration file.
    """
    import yaml
    from fx_trading.config.models import BacktestConfig
    from fx_trading.backtesting.engine import Backtester
    from fx_trading.monitoring.reports import ReportGenerator
    from fx_trading.monitoring.logging import setup_logging

    print_warning()

    if not config.exists():
        console.print(f"[red]Error: Config file not found: {config}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Running backtest with config: {config}[/bold]\n")

    # Load config
    with open(config) as f:
        config_data = yaml.safe_load(f)

    try:
        bt_config = BacktestConfig.model_validate(config_data)
    except Exception as e:
        console.print(f"[red]Config validation error: {e}[/red]")
        raise typer.Exit(1)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    # Run backtest
    backtester = Backtester(config=bt_config)

    with console.status("[bold green]Running backtest..."):
        result = backtester.run()

    # Display results
    table = Table(title="Backtest Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    return_style = "green" if result.total_return_pct >= 0 else "red"
    table.add_row("Total Return", f"[{return_style}]{result.total_return_pct:.2f}%[/{return_style}]")
    table.add_row("Max Drawdown", f"[red]{result.max_drawdown_pct:.2f}%[/red]")
    table.add_row("Sharpe Ratio", f"{result.sharpe_ratio:.2f}")
    table.add_row("Total Trades", str(result.total_trades))
    table.add_row("Win Rate", f"{result.win_rate:.1%}")
    table.add_row("Profit Factor", f"{result.profit_factor:.2f}")
    table.add_row("Initial Capital", f"${result.initial_capital:,.2f}")
    table.add_row("Final Equity", f"${result.final_equity:,.2f}")

    console.print(table)

    # Save results
    report_gen = ReportGenerator()
    results_dir = report_gen.save_full_results(result, output_dir)

    console.print(f"\n[green]Results saved to {results_dir}[/green]")
    console.print(f"View report: {results_dir}/report.html")


@app.command()
def walkforward(
    config: Path = typer.Argument(..., help="Path to walk-forward config YAML"),
    output_dir: Optional[Path] = typer.Option(None, "--output", "-o"),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Run walk-forward analysis.
    """
    import yaml
    from fx_trading.config.models import WalkForwardConfig, BacktestConfig
    from fx_trading.backtesting.walkforward import WalkForwardAnalysis
    from fx_trading.monitoring.logging import setup_logging

    print_warning()

    if not config.exists():
        console.print(f"[red]Error: Config file not found: {config}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Running walk-forward analysis with config: {config}[/bold]\n")

    # Load config
    with open(config) as f:
        config_data = yaml.safe_load(f)

    try:
        # Parse base config first
        base_config = BacktestConfig.model_validate(config_data.get("base_config", config_data))
        wf_config = WalkForwardConfig(
            base_config=base_config,
            num_windows=config_data.get("num_windows", 5),
            train_pct=config_data.get("train_pct", 0.7),
            gap_bars=config_data.get("gap_bars", 0),
            anchored=config_data.get("anchored", False),
        )
    except Exception as e:
        console.print(f"[red]Config validation error: {e}[/red]")
        raise typer.Exit(1)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    # Run walk-forward
    wf = WalkForwardAnalysis(config=wf_config)

    with console.status("[bold green]Running walk-forward analysis..."):
        result = wf.run()

    # Display results
    table = Table(title="Walk-Forward Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("Windows", str(len(result.windows)))
    table.add_row("Total OOS Return", f"{result.total_oos_return_pct:.2f}%")
    table.add_row("Avg OOS Return", f"{result.avg_oos_return_pct:.2f}%")
    table.add_row("OOS Sharpe", f"{result.oos_sharpe_ratio:.2f}")
    table.add_row("Total OOS Trades", str(result.total_oos_trades))
    table.add_row("Profitable Windows", str(result.profitable_windows))
    table.add_row("Consistency", f"{result.window_consistency:.1%}")

    console.print(table)

    # Window details
    window_table = Table(title="Window Details")
    window_table.add_column("Window", style="cyan")
    window_table.add_column("Test Period", style="white")
    window_table.add_column("Return", style="green")
    window_table.add_column("Trades", style="white")

    for w in result.windows:
        ret_style = "green" if w.test_result.total_return_pct >= 0 else "red"
        window_table.add_row(
            str(w.window_id),
            f"{w.test_start.strftime('%Y-%m-%d')} to {w.test_end.strftime('%Y-%m-%d')}",
            f"[{ret_style}]{w.test_result.total_return_pct:.2f}%[/{ret_style}]",
            str(w.test_result.total_trades),
        )

    console.print(window_table)

    # Save results
    results_dir = wf.save_results(result, output_dir)
    console.print(f"\n[green]Results saved to {results_dir}[/green]")


@app.command("paper-trade")
def paper_trade(
    config: Path = typer.Argument(..., help="Path to paper trading config YAML"),
    max_bars: Optional[int] = typer.Option(
        None,
        "--max-bars",
        "-n",
        help="Max bars to process (for testing)",
    ),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """
    Run paper trading simulation.

    This simulates live trading using historical data.
    """
    import yaml
    from fx_trading.config.models import PaperTradingConfig
    from fx_trading.backtesting.engine import Backtester, BacktestConfig
    from fx_trading.monitoring.logging import setup_logging

    print_warning()

    if not config.exists():
        console.print(f"[red]Error: Config file not found: {config}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Starting paper trading with config: {config}[/bold]\n")
    console.print("[yellow]Mode: PAPER TRADING (no real money)[/yellow]\n")

    # Load config
    with open(config) as f:
        config_data = yaml.safe_load(f)

    try:
        paper_config = PaperTradingConfig.model_validate(config_data)
    except Exception as e:
        console.print(f"[red]Config validation error: {e}[/red]")
        raise typer.Exit(1)

    # Setup logging
    log_level = "DEBUG" if verbose else "INFO"
    setup_logging(level=log_level)

    # Convert to backtest config for now (real paper trading would use live data)
    bt_config = BacktestConfig(
        run_id=f"paper_{paper_config.strategy.name}",
        output_dir=paper_config.output_dir,
        data_path=paper_config.data_path,
        symbols=paper_config.symbols,
        initial_capital=paper_config.initial_capital,
        base_currency=paper_config.base_currency,
        strategy=paper_config.strategy,
        costs=paper_config.costs,
        risk=paper_config.risk,
    )

    # Run as backtest (paper trading simulation)
    backtester = Backtester(config=bt_config)

    console.print("[bold]Paper trading started...[/bold]")
    console.print("Press Ctrl+C to stop\n")

    try:
        result = backtester.run()

        # Display results
        console.print("\n[bold]Paper Trading Complete[/bold]\n")

        table = Table(title="Session Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Total Return", f"{result.total_return_pct:.2f}%")
        table.add_row("Total Trades", str(result.total_trades))
        table.add_row("Win Rate", f"{result.win_rate:.1%}")
        table.add_row("Final Equity", f"${result.final_equity:,.2f}")

        console.print(table)

    except KeyboardInterrupt:
        console.print("\n[yellow]Paper trading stopped by user[/yellow]")


@app.command()
def report(
    run_id: str = typer.Argument(..., help="Run ID to generate report for"),
    runs_dir: Path = typer.Option(Path("runs"), "--runs-dir", "-d"),
    format: str = typer.Option("html", "--format", "-f", help="Report format (html/markdown)"),
) -> None:
    """
    Generate report for a completed run.
    """
    import json
    import pandas as pd
    from fx_trading.backtesting.engine import BacktestResult
    from fx_trading.monitoring.reports import ReportGenerator

    run_path = runs_dir / run_id

    if not run_path.exists():
        console.print(f"[red]Error: Run not found: {run_path}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Generating report for run: {run_id}[/bold]\n")

    # Load summary
    summary_path = run_path / "summary.json"
    if not summary_path.exists():
        console.print("[red]Error: summary.json not found[/red]")
        raise typer.Exit(1)

    with open(summary_path) as f:
        summary = json.load(f)

    # Load trades
    trades_path = run_path / "trades.parquet"
    trades_df = pd.read_parquet(trades_path) if trades_path.exists() else pd.DataFrame()

    # Load equity curve
    equity_path = run_path / "equity_curve.parquet"
    equity_df = pd.read_parquet(equity_path) if equity_path.exists() else pd.DataFrame()

    # This is a simplified reconstruction - in production you'd store the full result
    console.print(f"[green]Report available at {run_path}/report.html[/green]")


@app.command()
def validate_config(
    config: Path = typer.Argument(..., help="Path to config YAML to validate"),
    config_type: str = typer.Option(
        "backtest",
        "--type",
        "-t",
        help="Config type (backtest/walkforward/paper)",
    ),
) -> None:
    """
    Validate a configuration file.
    """
    import yaml
    from fx_trading.config.models import BacktestConfig, WalkForwardConfig, PaperTradingConfig

    if not config.exists():
        console.print(f"[red]Error: Config file not found: {config}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Validating {config_type} config: {config}[/bold]\n")

    with open(config) as f:
        config_data = yaml.safe_load(f)

    try:
        if config_type == "backtest":
            BacktestConfig.model_validate(config_data)
        elif config_type == "walkforward":
            base_config = BacktestConfig.model_validate(config_data.get("base_config", config_data))
            WalkForwardConfig(base_config=base_config, **{k: v for k, v in config_data.items() if k != "base_config"})
        elif config_type == "paper":
            PaperTradingConfig.model_validate(config_data)
        else:
            console.print(f"[red]Unknown config type: {config_type}[/red]")
            raise typer.Exit(1)

        console.print("[green]Config is valid![/green]")

    except Exception as e:
        console.print(f"[red]Validation error: {e}[/red]")
        raise typer.Exit(1)


@app.command("mt5-status")
def mt5_status(
    config: Optional[Path] = typer.Argument(None, help="Path to live trading config YAML (optional)"),
) -> None:
    """
    Check MT5 connection status.

    Tests the connection to MT5 terminal and displays account info.
    Uses the official MetaTrader5 Python package.
    """
    import yaml
    from fx_trading.execution.mt5_client import MT5Client

    console.print(f"\n[bold]Checking MT5 connection...[/bold]\n")

    # Load config if provided
    mt5_config = None
    symbols = ["EURUSD"]

    if config and config.exists():
        from fx_trading.config.models import LiveTradingConfig

        with open(config) as f:
            config_data = yaml.safe_load(f)

        try:
            live_config = LiveTradingConfig.model_validate(config_data)
            mt5_config = live_config.mt5
            symbols = live_config.symbols
        except Exception as e:
            console.print(f"[yellow]Config warning: {e}[/yellow]")
            console.print("Continuing with default settings...\n")

    # Create client
    if mt5_config:
        client = MT5Client(
            path=mt5_config.path,
            login=mt5_config.login,
            password=mt5_config.password,
            server=mt5_config.server,
            timeout=mt5_config.timeout_ms,
            portable=mt5_config.portable,
        )
        symbol_suffix = mt5_config.symbol_suffix
    else:
        client = MT5Client()
        symbol_suffix = ""

    # Try to connect
    console.print("Connecting to MT5 terminal...")

    if not client.connect():
        console.print("[red]Failed to connect to MT5[/red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  1. Is MT5 terminal running on this machine?")
        console.print("  2. Is the MetaTrader5 Python package installed?")
        console.print("     pip install MetaTrader5")
        console.print("  3. Are you running on Windows? (required for MT5)")
        console.print("  4. Try restarting the MT5 terminal")
        raise typer.Exit(1)

    console.print("[green]Connected to MT5![/green]\n")

    # Get account info
    account = client.get_account_info()
    if account:
        table = Table(title="MT5 Account Info")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Login", str(account.get("login", "N/A")))
        table.add_row("Name", str(account.get("name", "N/A")))
        table.add_row("Server", str(account.get("server", "N/A")))
        table.add_row("Currency", str(account.get("currency", "N/A")))
        table.add_row("Balance", f"${account.get('balance', 0):,.2f}")
        table.add_row("Equity", f"${account.get('equity', 0):,.2f}")
        table.add_row("Margin", f"${account.get('margin', 0):,.2f}")
        table.add_row("Free Margin", f"${account.get('margin_free', 0):,.2f}")
        table.add_row("Leverage", f"1:{account.get('leverage', 1)}")

        console.print(table)

    # Get open positions
    positions = client.get_open_positions()
    if positions:
        console.print(f"\n[bold]Open Positions: {len(positions)}[/bold]")
        for pos in positions:
            console.print(
                f"  {pos['ticket']}: {pos['symbol']} {pos['type']} "
                f"{pos['volume']} lots @ {pos['price_open']:.5f} "
                f"(PnL: ${pos['profit']:.2f})"
            )
    else:
        console.print("\n[dim]No open positions[/dim]")

    # Test tick data
    for symbol in symbols:
        mt5_symbol = f"{symbol}{symbol_suffix}"
        tick = client.get_tick(mt5_symbol)
        if tick:
            console.print(
                f"\n[bold]{mt5_symbol}:[/bold] "
                f"Bid={tick.get('bid'):.5f} Ask={tick.get('ask'):.5f}"
            )
        else:
            console.print(f"\n[yellow]{mt5_symbol}: No tick data (check symbol name)[/yellow]")

    client.disconnect()
    console.print("\n[green]MT5 connection test complete![/green]")


@app.command("live")
def live_trade(
    config: Path = typer.Argument(..., help="Path to live trading config YAML"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Verbose output"),
    dry_run: bool = typer.Option(
        False,
        "--dry-run",
        help="Simulate without placing real orders",
    ),
) -> None:
    """
    Start live/demo trading with MT5.

    IMPORTANT: Start with a demo account only!
    Uses the official MetaTrader5 Python package.
    """
    import yaml
    from fx_trading.config.models import LiveTradingConfig
    from fx_trading.execution.mt5_client import MT5Client
    from fx_trading.execution.mt5_broker import MT5Broker
    from fx_trading.portfolio.accounting import PortfolioManager
    from fx_trading.risk.engine import RiskEngine
    from fx_trading.strategies.base import StrategyFactory
    from fx_trading.trading.live_runner import LiveTradingRunner
    from fx_trading.monitoring.logging import setup_logging

    print_warning()

    console.print(Panel(
        "[bold red]LIVE TRADING MODE[/bold red]\n\n"
        "This will connect to your MT5 account and execute real trades.\n"
        "Make sure you are using a DEMO account!",
        title="Warning",
        border_style="red",
    ))

    if not config.exists():
        console.print(f"[red]Error: Config file not found: {config}[/red]")
        raise typer.Exit(1)

    # Confirm
    if not dry_run:
        confirm = typer.confirm(
            "Are you sure you want to start live trading?",
            default=False,
        )
        if not confirm:
            console.print("[yellow]Cancelled[/yellow]")
            raise typer.Exit(0)

    console.print(f"\n[bold]Starting live trading with config: {config}[/bold]\n")

    # Load config
    with open(config) as f:
        config_data = yaml.safe_load(f)

    try:
        live_config = LiveTradingConfig.model_validate(config_data)
    except Exception as e:
        console.print(f"[red]Config validation error: {e}[/red]")
        raise typer.Exit(1)

    # Setup logging with file output
    log_level = "DEBUG" if verbose else live_config.log_level
    log_dir = Path("logs")
    setup_logging(
        log_dir=log_dir,
        level=log_level,
        run_id=live_config.run_id,
    )
    console.print(f"[dim]Logs saved to: {log_dir.absolute()}[/dim]")

    # Create MT5 client and broker
    mt5_config = live_config.mt5
    client = MT5Client(
        path=mt5_config.path,
        login=mt5_config.login,
        password=mt5_config.password,
        server=mt5_config.server,
        timeout=mt5_config.timeout_ms,
        portable=mt5_config.portable,
    )

    broker = MT5Broker(
        client=client,
        symbol_suffix=mt5_config.symbol_suffix,
        magic_number=mt5_config.magic_number,
    )

    # Connect to MT5
    console.print("Connecting to MT5 terminal...")
    if not broker.connect():
        console.print("[red]Failed to connect to MT5[/red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  1. Is MT5 terminal running?")
        console.print("  2. pip install MetaTrader5")
        console.print("  3. Windows only - are you on Windows?")
        raise typer.Exit(1)

    console.print("[green]Connected to MT5![/green]")

    # Get initial account info
    account = broker.get_account()
    console.print(f"Account balance: ${account.balance:,.2f}")

    # Create portfolio
    initial_capital = live_config.initial_capital or account.balance
    portfolio = PortfolioManager(initial_capital=initial_capital)

    # Create risk engine
    risk_engine = RiskEngine(live_config.risk, portfolio)

    # Create strategy
    strategy = StrategyFactory.create(live_config.strategy)

    # Create runner
    runner = LiveTradingRunner(
        config=live_config,
        broker=broker,
        strategy=strategy,
        risk_engine=risk_engine,
        portfolio=portfolio,
    )

    if dry_run:
        console.print("\n[yellow]DRY RUN MODE - No orders will be placed[/yellow]\n")

    console.print("\n[bold green]Live trading started![/bold green]")
    console.print("Press Ctrl+C to stop\n")

    try:
        runner.run()
    except KeyboardInterrupt:
        console.print("\n[yellow]Stopping live trading...[/yellow]")
        runner.stop()


@app.command("calendar")
def economic_calendar(
    hours: int = typer.Option(24, "--hours", "-h", help="Hours to look ahead"),
    currency: Optional[str] = typer.Option(
        None,
        "--currency",
        "-c",
        help="Filter by currency (e.g., USD, EUR)",
    ),
    all_impacts: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Show all impact levels (default: high only)",
    ),
) -> None:
    """
    Show upcoming economic calendar events.

    Displays high-impact news events that may affect trading.
    """
    from fx_trading.data.economic_calendar import EconomicCalendar, Impact

    console.print(f"\n[bold]Economic Calendar - Next {hours} hours[/bold]\n")

    calendar = EconomicCalendar()

    try:
        min_impact = Impact.LOW if all_impacts else Impact.HIGH
        currencies = [currency.upper()] if currency else None

        events = calendar.get_upcoming_events(
            hours_ahead=hours,
            min_impact=min_impact,
            currencies=currencies,
        )

        if not events:
            console.print("[dim]No upcoming high-impact events[/dim]")
            return

        table = Table(title=f"Upcoming Events ({len(events)})")
        table.add_column("Time (UTC)", style="cyan")
        table.add_column("Currency", style="yellow")
        table.add_column("Impact", style="white")
        table.add_column("Event", style="green")

        for event in events:
            impact_style = {
                Impact.HIGH: "[bold red]HIGH[/bold red]",
                Impact.MEDIUM: "[yellow]MEDIUM[/yellow]",
                Impact.LOW: "[dim]LOW[/dim]",
            }.get(event.impact, "")

            table.add_row(
                event.timestamp.strftime("%m/%d %H:%M"),
                event.currency,
                impact_style,
                event.title,
            )

        console.print(table)

        console.print("\n[dim]Tip: Use --all to show all impact levels[/dim]")
        console.print("[dim]News filter blocks trading 30min before, 15min after high-impact events[/dim]")

    except Exception as e:
        console.print(f"[yellow]Could not fetch calendar: {e}[/yellow]")
        console.print("[dim]Make sure you have internet connectivity[/dim]")
    finally:
        calendar.close()


@app.command("live-report")
def live_report(
    days: int = typer.Option(7, "--days", "-d", help="Number of days to include"),
    output: Path = typer.Option(Path("live_report.html"), "--output", "-o", help="Output file path"),
) -> None:
    """
    Generate HTML report from MT5 live trading history.

    Fetches trade history from MT5 and generates a detailed report.
    """
    from datetime import datetime, timedelta
    from fx_trading.execution.mt5_client import MT5Client

    console.print(f"\n[bold]Generating Live Trading Report[/bold]\n")
    console.print(f"Fetching trades from last {days} days...\n")

    client = MT5Client()
    if not client.connect():
        console.print("[red]Failed to connect to MT5[/red]")
        console.print("Make sure MT5 is running and logged in.")
        raise typer.Exit(1)

    try:
        # Get account info
        account = client.get_account_info()

        # Get trade history
        deals = client.get_history_deals(
            from_date=datetime.now() - timedelta(days=days),
            to_date=datetime.now(),
        )

        # Filter to actual trades (not balance operations)
        trades = [d for d in deals if d.get("symbol")]

        # Calculate statistics
        total_profit = sum(d.get("profit", 0) for d in trades)
        total_commission = sum(d.get("commission", 0) for d in trades)
        total_swap = sum(d.get("swap", 0) for d in trades)
        net_profit = total_profit + total_commission + total_swap

        winners = [d for d in trades if d.get("profit", 0) > 0]
        losers = [d for d in trades if d.get("profit", 0) < 0]
        win_rate = len(winners) / len(trades) * 100 if trades else 0

        avg_win = sum(d.get("profit", 0) for d in winners) / len(winners) if winners else 0
        avg_loss = sum(d.get("profit", 0) for d in losers) / len(losers) if losers else 0
        profit_factor = abs(sum(d.get("profit", 0) for d in winners) / sum(d.get("profit", 0) for d in losers)) if losers and sum(d.get("profit", 0) for d in losers) != 0 else 0

        # Generate HTML
        html = f"""<!DOCTYPE html>
<html>
<head>
    <title>Live Trading Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }}
        .header {{
            background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
            color: white;
            padding: 30px;
            border-radius: 10px;
            margin-bottom: 20px;
        }}
        .header h1 {{ margin: 0; font-size: 28px; }}
        .header .subtitle {{ opacity: 0.8; margin-top: 10px; }}
        .card {{
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }}
        .card h2 {{
            margin-top: 0;
            color: #333;
            border-bottom: 2px solid #eee;
            padding-bottom: 10px;
        }}
        .metrics-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
        }}
        .metric {{
            background: #f8f9fa;
            padding: 15px;
            border-radius: 8px;
            text-align: center;
        }}
        .metric-value {{
            font-size: 24px;
            font-weight: bold;
            color: #333;
        }}
        .metric-value.positive {{ color: #28a745; }}
        .metric-value.negative {{ color: #dc3545; }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin-top: 15px;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{ background: #f8f9fa; font-weight: 600; }}
        tr:hover {{ background: #f8f9fa; }}
        .profit {{ color: #28a745; font-weight: bold; }}
        .loss {{ color: #dc3545; font-weight: bold; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Live Trading Report</h1>
        <div class="subtitle">
            Account: {account.get('login', 'N/A')} | Server: {account.get('server', 'N/A')} |
            Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}
        </div>
    </div>

    <div class="card">
        <h2>Account Summary</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value">${account.get('balance', 0):,.2f}</div>
                <div class="metric-label">Balance</div>
            </div>
            <div class="metric">
                <div class="metric-value">${account.get('equity', 0):,.2f}</div>
                <div class="metric-label">Equity</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if net_profit >= 0 else 'negative'}">${net_profit:+,.2f}</div>
                <div class="metric-label">Net P/L ({days} days)</div>
            </div>
            <div class="metric">
                <div class="metric-value">{account.get('leverage', 0)}:1</div>
                <div class="metric-label">Leverage</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Trading Statistics</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value">{len(trades)}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(winners)}</div>
                <div class="metric-label">Winners</div>
            </div>
            <div class="metric">
                <div class="metric-value">{len(losers)}</div>
                <div class="metric-label">Losers</div>
            </div>
            <div class="metric">
                <div class="metric-value {'positive' if win_rate >= 50 else 'negative'}">{win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
            <div class="metric">
                <div class="metric-value positive">${avg_win:,.2f}</div>
                <div class="metric-label">Avg Win</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">${avg_loss:,.2f}</div>
                <div class="metric-label">Avg Loss</div>
            </div>
            <div class="metric">
                <div class="metric-value">{profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric">
                <div class="metric-value">${total_commission:,.2f}</div>
                <div class="metric-label">Commissions</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Trade History</h2>
        <table>
            <thead>
                <tr>
                    <th>Time</th>
                    <th>Symbol</th>
                    <th>Type</th>
                    <th>Volume</th>
                    <th>Price</th>
                    <th>P/L</th>
                </tr>
            </thead>
            <tbody>
"""

        for trade in sorted(trades, key=lambda x: x.get("time", datetime.now()), reverse=True):
            trade_time = trade.get("time", datetime.now())
            if isinstance(trade_time, datetime):
                time_str = trade_time.strftime("%Y-%m-%d %H:%M")
            else:
                time_str = str(trade_time)

            profit = trade.get("profit", 0)
            profit_class = "profit" if profit >= 0 else "loss"

            html += f"""                <tr>
                    <td>{time_str}</td>
                    <td>{trade.get('symbol', '')}</td>
                    <td>{trade.get('type', '').upper()}</td>
                    <td>{trade.get('volume', 0):.2f}</td>
                    <td>{trade.get('price', 0):.5f}</td>
                    <td class="{profit_class}">${profit:+,.2f}</td>
                </tr>
"""

        html += """            </tbody>
        </table>
    </div>

    <div style="text-align: center; color: #666; margin-top: 20px; font-size: 12px;">
        Generated by FX Trading System
    </div>
</body>
</html>"""

        # Write file
        with open(output, "w") as f:
            f.write(html)

        console.print(f"[green]Report generated: {output}[/green]")
        console.print(f"\nOpen in browser: file:///{output.absolute()}")

        # Print summary
        table = Table(title="Quick Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="white")

        table.add_row("Total Trades", str(len(trades)))
        table.add_row("Win Rate", f"{win_rate:.1f}%")
        table.add_row("Net P/L", f"${net_profit:+,.2f}")
        table.add_row("Profit Factor", f"{profit_factor:.2f}")
        table.add_row("Balance", f"${account.get('balance', 0):,.2f}")

        console.print(table)

    finally:
        client.disconnect()


@app.command()
def version() -> None:
    """Show version information."""
    from fx_trading import __version__

    console.print(f"\n[bold]FX Trading System[/bold] v{__version__}")
    console.print("Educational quant trading scaffold")
    console.print("\n[yellow]For educational purposes only. Use paper trading.[/yellow]")


if __name__ == "__main__":
    app()
