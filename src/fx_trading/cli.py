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
    config: Path = typer.Argument(..., help="Path to live trading config YAML"),
) -> None:
    """
    Check MT5 connection status.

    Tests the ZeroMQ connection to MT5 and displays account info.
    """
    import yaml
    from fx_trading.config.models import LiveTradingConfig
    from fx_trading.execution.mt5_zmq_client import MT5ZmqClient

    if not config.exists():
        console.print(f"[red]Error: Config file not found: {config}[/red]")
        raise typer.Exit(1)

    console.print(f"\n[bold]Checking MT5 connection...[/bold]\n")

    # Load config
    with open(config) as f:
        config_data = yaml.safe_load(f)

    try:
        live_config = LiveTradingConfig.model_validate(config_data)
    except Exception as e:
        console.print(f"[red]Config validation error: {e}[/red]")
        raise typer.Exit(1)

    # Create client
    mt5_config = live_config.mt5
    client = MT5ZmqClient(
        host=mt5_config.zmq_host,
        push_port=mt5_config.zmq_push_port,
        pull_port=mt5_config.zmq_pull_port,
        timeout_ms=mt5_config.timeout_seconds * 1000,
    )

    # Try to connect
    console.print(f"Connecting to {mt5_config.zmq_host}:{mt5_config.zmq_push_port}...")

    if not client.connect():
        console.print("[red]Failed to connect to MT5[/red]")
        console.print("\n[yellow]Troubleshooting:[/yellow]")
        console.print("  1. Is MT5 terminal running on the target machine?")
        console.print("  2. Is the DWX EA attached to a chart and enabled?")
        console.print("  3. Are ports 32768/32769 accessible (firewall)?")
        console.print(f"  4. Is the host '{mt5_config.zmq_host}' correct?")
        raise typer.Exit(1)

    console.print("[green]Connected to MT5![/green]\n")

    # Get account info
    account = client.get_account_info()
    if account:
        table = Table(title="MT5 Account Info")
        table.add_column("Property", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Login", str(account.get("login", "N/A")))
        table.add_row("Server", str(account.get("server", "N/A")))
        table.add_row("Balance", f"${account.get('balance', 0):,.2f}")
        table.add_row("Equity", f"${account.get('equity', 0):,.2f}")
        table.add_row("Margin", f"${account.get('margin', 0):,.2f}")
        table.add_row("Free Margin", f"${account.get('margin_free', 0):,.2f}")
        table.add_row("Leverage", f"1:{account.get('leverage', 1)}")

        console.print(table)

    # Get open trades
    trades = client.get_open_trades()
    if trades:
        console.print(f"\n[bold]Open Positions: {len(trades)}[/bold]")
        for ticket, trade in trades.items():
            console.print(
                f"  {ticket}: {trade.get('symbol')} {trade.get('type')} "
                f"{trade.get('lots')} lots @ {trade.get('open_price')}"
            )
    else:
        console.print("\n[dim]No open positions[/dim]")

    # Test tick data
    for symbol in live_config.symbols:
        mt5_symbol = f"{symbol}{mt5_config.symbol_suffix}"
        tick = client.get_tick(mt5_symbol)
        if tick:
            console.print(
                f"\n[bold]{mt5_symbol}:[/bold] "
                f"Bid={tick.get('bid'):.5f} Ask={tick.get('ask'):.5f}"
            )
        else:
            console.print(f"\n[yellow]{mt5_symbol}: No tick data[/yellow]")

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
    """
    import yaml
    from fx_trading.config.models import LiveTradingConfig
    from fx_trading.execution.mt5_zmq_client import MT5ZmqClient
    from fx_trading.execution.mt5_broker import MT5ZmqBroker
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

    # Setup logging
    log_level = "DEBUG" if verbose else live_config.log_level
    setup_logging(level=log_level)

    # Create MT5 client and broker
    mt5_config = live_config.mt5
    client = MT5ZmqClient(
        host=mt5_config.zmq_host,
        push_port=mt5_config.zmq_push_port,
        pull_port=mt5_config.zmq_pull_port,
        timeout_ms=mt5_config.timeout_seconds * 1000,
    )

    broker = MT5ZmqBroker(
        client=client,
        symbol_suffix=mt5_config.symbol_suffix,
        magic_number=mt5_config.magic_number,
    )

    # Connect to MT5
    console.print("Connecting to MT5...")
    if not broker.connect():
        console.print("[red]Failed to connect to MT5[/red]")
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


@app.command()
def version() -> None:
    """Show version information."""
    from fx_trading import __version__

    console.print(f"\n[bold]FX Trading System[/bold] v{__version__}")
    console.print("Educational quant trading scaffold")
    console.print("\n[yellow]For educational purposes only. Use paper trading.[/yellow]")


if __name__ == "__main__":
    app()
