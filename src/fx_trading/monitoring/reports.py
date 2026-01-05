"""
Report generation for backtest and trading results.

Generates HTML and markdown reports with charts.
"""

import json
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger

from fx_trading.backtesting.engine import BacktestResult


class ReportGenerator:
    """
    Generate reports from backtest/trading results.

    Supports HTML and markdown output with optional charts.
    """

    HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Trading Report - {run_id}</title>
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
        .header h1 {{
            margin: 0;
            font-size: 28px;
        }}
        .header .subtitle {{
            opacity: 0.8;
            margin-top: 10px;
        }}
        .warning {{
            background: #fff3cd;
            border: 1px solid #ffc107;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
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
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
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
        .metric-value.positive {{
            color: #28a745;
        }}
        .metric-value.negative {{
            color: #dc3545;
        }}
        .metric-label {{
            font-size: 12px;
            color: #666;
            margin-top: 5px;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
        }}
        th, td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }}
        th {{
            background: #f8f9fa;
            font-weight: 600;
        }}
        tr:hover {{
            background: #f8f9fa;
        }}
        .pnl-positive {{
            color: #28a745;
        }}
        .pnl-negative {{
            color: #dc3545;
        }}
        .chart-container {{
            margin: 20px 0;
        }}
        .footer {{
            text-align: center;
            color: #666;
            font-size: 12px;
            margin-top: 40px;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>Trading Report</h1>
        <div class="subtitle">Run ID: {run_id} | Generated: {generated_at}</div>
    </div>

    <div class="warning">
        <strong>Disclaimer:</strong> This is an educational trading system. Past performance
        does not guarantee future results. Trading forex involves significant risk of loss.
        Use paper trading only.
    </div>

    <div class="card">
        <h2>Performance Summary</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value {return_class}">{total_return:.2f}%</div>
                <div class="metric-label">Total Return</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">{max_drawdown:.2f}%</div>
                <div class="metric-label">Max Drawdown</div>
            </div>
            <div class="metric">
                <div class="metric-value">{sharpe:.2f}</div>
                <div class="metric-label">Sharpe Ratio</div>
            </div>
            <div class="metric">
                <div class="metric-value">{profit_factor:.2f}</div>
                <div class="metric-label">Profit Factor</div>
            </div>
            <div class="metric">
                <div class="metric-value">{total_trades}</div>
                <div class="metric-label">Total Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">{win_rate:.1f}%</div>
                <div class="metric-label">Win Rate</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Account Summary</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value">${initial_capital:,.2f}</div>
                <div class="metric-label">Initial Capital</div>
            </div>
            <div class="metric">
                <div class="metric-value {final_class}">${final_equity:,.2f}</div>
                <div class="metric-label">Final Equity</div>
            </div>
            <div class="metric">
                <div class="metric-value {pnl_class}">${net_pnl:,.2f}</div>
                <div class="metric-label">Net P&L</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Cost Analysis</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value negative">${total_commission:,.2f}</div>
                <div class="metric-label">Total Commission</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">${total_slippage:,.2f}</div>
                <div class="metric-label">Total Slippage</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">${total_spread:,.2f}</div>
                <div class="metric-label">Est. Spread Cost</div>
            </div>
        </div>
    </div>

    <div class="card">
        <h2>Trade Statistics</h2>
        <div class="metrics-grid">
            <div class="metric">
                <div class="metric-value">{winning_trades}</div>
                <div class="metric-label">Winning Trades</div>
            </div>
            <div class="metric">
                <div class="metric-value">{losing_trades}</div>
                <div class="metric-label">Losing Trades</div>
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
                <div class="metric-value positive">${max_win:,.2f}</div>
                <div class="metric-label">Largest Win</div>
            </div>
            <div class="metric">
                <div class="metric-value negative">${max_loss:,.2f}</div>
                <div class="metric-label">Largest Loss</div>
            </div>
        </div>
    </div>

    {trades_table}

    <div class="footer">
        <p>Generated by FX Trading System | Educational Use Only</p>
    </div>
</body>
</html>
"""

    def __init__(self, output_dir: Optional[Path] = None):
        """
        Initialize report generator.

        Args:
            output_dir: Output directory for reports
        """
        self.output_dir = Path(output_dir) if output_dir else Path("runs")

    def generate_html_report(
        self,
        result: BacktestResult,
        include_trades: bool = True,
        include_charts: bool = True,
    ) -> str:
        """
        Generate HTML report from backtest result.

        Args:
            result: BacktestResult to report on
            include_trades: Include trades table
            include_charts: Include equity chart

        Returns:
            HTML string
        """
        # Calculate derived values
        net_pnl = result.final_equity - result.initial_capital
        return_class = "positive" if result.total_return_pct >= 0 else "negative"
        final_class = "positive" if result.final_equity >= result.initial_capital else "negative"
        pnl_class = "positive" if net_pnl >= 0 else "negative"

        # Generate trades table
        trades_table = ""
        if include_trades and not result.trades_df.empty:
            trades_table = self._generate_trades_table(result.trades_df)

        html = self.HTML_TEMPLATE.format(
            run_id=result.run_id,
            generated_at=datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
            total_return=result.total_return_pct,
            return_class=return_class,
            max_drawdown=result.max_drawdown_pct,
            sharpe=result.sharpe_ratio,
            profit_factor=result.profit_factor if result.profit_factor != float("inf") else 999.99,
            total_trades=result.total_trades,
            win_rate=result.win_rate * 100,
            initial_capital=result.initial_capital,
            final_equity=result.final_equity,
            final_class=final_class,
            net_pnl=net_pnl,
            pnl_class=pnl_class,
            total_commission=result.total_commission,
            total_slippage=result.total_slippage,
            total_spread=result.total_spread_cost,
            winning_trades=result.winning_trades,
            losing_trades=result.losing_trades,
            avg_win=result.avg_win,
            avg_loss=abs(result.avg_loss),
            max_win=result.max_win,
            max_loss=abs(result.max_loss),
            trades_table=trades_table,
        )

        return html

    def _generate_trades_table(self, trades_df: pd.DataFrame, max_rows: int = 50) -> str:
        """Generate HTML table for trades."""
        if trades_df.empty:
            return ""

        # Take recent trades
        df = trades_df.tail(max_rows).copy()

        rows = []
        for _, trade in df.iterrows():
            pnl_class = "pnl-positive" if trade["net_pnl"] >= 0 else "pnl-negative"
            rows.append(f"""
            <tr>
                <td>{trade['entry_time']}</td>
                <td>{trade['symbol']}</td>
                <td>{trade['side']}</td>
                <td>{trade['size']:.2f}</td>
                <td>{trade['entry_price']:.5f}</td>
                <td>{trade['exit_price']:.5f}</td>
                <td class="{pnl_class}">${trade['net_pnl']:.2f}</td>
                <td>{trade['exit_reason']}</td>
            </tr>
            """)

        return f"""
        <div class="card">
            <h2>Recent Trades (Last {len(df)})</h2>
            <table>
                <thead>
                    <tr>
                        <th>Entry Time</th>
                        <th>Symbol</th>
                        <th>Side</th>
                        <th>Size</th>
                        <th>Entry</th>
                        <th>Exit</th>
                        <th>P&L</th>
                        <th>Exit Reason</th>
                    </tr>
                </thead>
                <tbody>
                    {''.join(rows)}
                </tbody>
            </table>
        </div>
        """

    def generate_markdown_report(self, result: BacktestResult) -> str:
        """
        Generate markdown report.

        Args:
            result: BacktestResult

        Returns:
            Markdown string
        """
        net_pnl = result.final_equity - result.initial_capital

        md = f"""# Trading Report

**Run ID:** {result.run_id}
**Generated:** {datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")}

---

## Performance Summary

| Metric | Value |
|--------|-------|
| Total Return | {result.total_return_pct:.2f}% |
| Max Drawdown | {result.max_drawdown_pct:.2f}% |
| Sharpe Ratio | {result.sharpe_ratio:.2f} |
| Profit Factor | {result.profit_factor:.2f} |
| Total Trades | {result.total_trades} |
| Win Rate | {result.win_rate:.1%} |

## Account Summary

| Metric | Value |
|--------|-------|
| Initial Capital | ${result.initial_capital:,.2f} |
| Final Equity | ${result.final_equity:,.2f} |
| Net P&L | ${net_pnl:,.2f} |

## Cost Analysis

| Cost Type | Amount |
|-----------|--------|
| Commission | ${result.total_commission:,.2f} |
| Slippage | ${result.total_slippage:,.2f} |
| Spread (est.) | ${result.total_spread_cost:,.2f} |

## Trade Statistics

| Metric | Value |
|--------|-------|
| Winning Trades | {result.winning_trades} |
| Losing Trades | {result.losing_trades} |
| Average Win | ${result.avg_win:,.2f} |
| Average Loss | ${abs(result.avg_loss):,.2f} |
| Largest Win | ${result.max_win:,.2f} |
| Largest Loss | ${abs(result.max_loss):,.2f} |
| Expectancy | ${result.expectancy:,.2f} |

---

*Disclaimer: This is an educational trading system. Past performance does not guarantee future results.*
"""
        return md

    def save_report(
        self,
        result: BacktestResult,
        output_dir: Optional[Path] = None,
        format: str = "html",
    ) -> Path:
        """
        Save report to file.

        Args:
            result: BacktestResult
            output_dir: Output directory
            format: "html" or "markdown"

        Returns:
            Path to saved report
        """
        output_dir = output_dir or self.output_dir / result.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        if format == "html":
            content = self.generate_html_report(result)
            path = output_dir / "report.html"
        else:
            content = self.generate_markdown_report(result)
            path = output_dir / "report.md"

        with open(path, "w") as f:
            f.write(content)

        logger.info(f"Report saved to {path}")
        return path

    def save_full_results(
        self,
        result: BacktestResult,
        output_dir: Optional[Path] = None,
    ) -> Path:
        """
        Save complete results including data files.

        Args:
            result: BacktestResult
            output_dir: Output directory

        Returns:
            Path to results directory
        """
        output_dir = output_dir or self.output_dir / result.run_id
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save HTML report
        self.save_report(result, output_dir, "html")

        # Save summary JSON
        summary_path = output_dir / "summary.json"
        with open(summary_path, "w") as f:
            json.dump(result.to_dict(), f, indent=2, default=str)

        # Save trades
        if not result.trades_df.empty:
            trades_path = output_dir / "trades.parquet"
            result.trades_df.to_parquet(trades_path)

        # Save equity curve
        if not result.equity_curve.empty:
            equity_path = output_dir / "equity_curve.parquet"
            result.equity_curve.to_parquet(equity_path)

        # Save signals log
        if result.signals_log:
            signals_path = output_dir / "signals.json"
            with open(signals_path, "w") as f:
                json.dump(result.signals_log, f, indent=2, default=str)

        # Save risk decisions
        if result.risk_decisions:
            risk_path = output_dir / "risk_decisions.json"
            with open(risk_path, "w") as f:
                json.dump(result.risk_decisions, f, indent=2, default=str)

        logger.info(f"Full results saved to {output_dir}")
        return output_dir

    def generate_equity_chart(
        self,
        equity_curve: pd.DataFrame,
        output_path: Optional[Path] = None,
    ) -> Optional[Path]:
        """
        Generate equity curve chart.

        Args:
            equity_curve: Equity curve DataFrame
            output_path: Output path for chart

        Returns:
            Path to chart file or None
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.dates as mdates

            if equity_curve.empty:
                return None

            fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

            # Equity curve
            ax1 = axes[0]
            ax1.plot(equity_curve.index, equity_curve["equity"], label="Equity", color="#2196F3")
            ax1.fill_between(equity_curve.index, equity_curve["equity"], alpha=0.3)
            ax1.set_ylabel("Equity ($)")
            ax1.set_title("Equity Curve")
            ax1.legend()
            ax1.grid(True, alpha=0.3)

            # Drawdown
            ax2 = axes[1]
            ax2.fill_between(
                equity_curve.index,
                -equity_curve["drawdown_pct"],
                0,
                color="#f44336",
                alpha=0.5,
            )
            ax2.set_ylabel("Drawdown (%)")
            ax2.set_xlabel("Date")
            ax2.set_title("Drawdown")
            ax2.grid(True, alpha=0.3)

            # Format x-axis
            ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m-%d"))
            plt.xticks(rotation=45)

            plt.tight_layout()

            if output_path:
                plt.savefig(output_path, dpi=150, bbox_inches="tight")
                plt.close()
                logger.info(f"Chart saved to {output_path}")
                return output_path
            else:
                plt.show()
                return None

        except ImportError:
            logger.warning("matplotlib not available for chart generation")
            return None
