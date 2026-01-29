#!/usr/bin/env python3
"""
Download historical M5 data from MT5 terminal.

Usage:
    python scripts/download_mt5_history.py --symbols EURUSD GBPUSD USDJPY XAUUSD --months 6
    python scripts/download_mt5_history.py --symbols EURUSD --start 2024-01-01 --end 2024-12-31
    python scripts/download_mt5_history.py --symbols EURUSD --timeframe M15 --months 12

Requirements:
    - MetaTrader5 package (Windows only): pip install MetaTrader5
    - MT5 terminal must be installed and logged in
"""

import argparse
import sys
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

try:
    import MetaTrader5 as mt5
except ImportError:
    print("ERROR: MetaTrader5 package not installed.")
    print("This script must be run on Windows with MT5 installed.")
    print("Install with: pip install MetaTrader5")
    sys.exit(1)


# MT5 timeframe mapping
TIMEFRAMES = {
    "M1": mt5.TIMEFRAME_M1,
    "M5": mt5.TIMEFRAME_M5,
    "M15": mt5.TIMEFRAME_M15,
    "M30": mt5.TIMEFRAME_M30,
    "H1": mt5.TIMEFRAME_H1,
    "H4": mt5.TIMEFRAME_H4,
    "D1": mt5.TIMEFRAME_D1,
}


def download_symbol_data(
    symbol: str,
    timeframe: str,
    start_date: datetime,
    end_date: datetime,
) -> pd.DataFrame | None:
    """Download historical data for a single symbol."""

    tf = TIMEFRAMES.get(timeframe)
    if tf is None:
        print(f"  ERROR: Unknown timeframe {timeframe}")
        return None

    # Request data from MT5
    rates = mt5.copy_rates_range(symbol, tf, start_date, end_date)

    if rates is None or len(rates) == 0:
        error = mt5.last_error()
        print(f"  ERROR: No data returned for {symbol}. MT5 error: {error}")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(rates)

    # Convert time column to datetime index
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df.set_index("time", inplace=True)
    df.index.name = "Datetime"

    # Rename columns to match expected format
    df = df.rename(columns={
        "open": "open",
        "high": "high",
        "low": "low",
        "close": "close",
        "tick_volume": "volume",
    })

    # Add symbol column
    df["symbol"] = symbol

    # Keep only required columns in correct order
    df = df[["open", "high", "low", "close", "volume", "symbol"]]

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Download historical data from MT5",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Download 6 months of M5 data for multiple symbols
    python scripts/download_mt5_history.py --symbols EURUSD GBPUSD USDJPY --months 6

    # Download specific date range
    python scripts/download_mt5_history.py --symbols EURUSD --start 2024-01-01 --end 2024-06-30

    # Download H1 data for 1 year
    python scripts/download_mt5_history.py --symbols XAUUSD --timeframe H1 --months 12

    # Download to custom directory
    python scripts/download_mt5_history.py --symbols EURUSD --months 3 --output data/mt5_history
        """
    )

    parser.add_argument(
        "--symbols",
        nargs="+",
        required=True,
        help="Symbols to download (e.g., EURUSD GBPUSD XAUUSD)",
    )
    parser.add_argument(
        "--timeframe",
        default="M5",
        choices=list(TIMEFRAMES.keys()),
        help="Timeframe (default: M5)",
    )
    parser.add_argument(
        "--months",
        type=int,
        default=None,
        help="Number of months to download (from today backwards)",
    )
    parser.add_argument(
        "--start",
        type=str,
        default=None,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end",
        type=str,
        default=None,
        help="End date (YYYY-MM-DD), defaults to today",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/sample",
        help="Output directory (default: data/sample)",
    )

    args = parser.parse_args()

    # Determine date range
    end_date = datetime.now()
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d")

    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d")
    elif args.months:
        start_date = end_date - timedelta(days=args.months * 30)
    else:
        print("ERROR: Specify either --months or --start date")
        sys.exit(1)

    print(f"MT5 Historical Data Downloader")
    print(f"=" * 50)
    print(f"Symbols:   {', '.join(args.symbols)}")
    print(f"Timeframe: {args.timeframe}")
    print(f"Period:    {start_date.date()} to {end_date.date()}")
    print(f"Output:    {args.output}")
    print()

    # Initialize MT5 connection
    print("Connecting to MT5...")
    if not mt5.initialize():
        print(f"ERROR: MT5 initialization failed: {mt5.last_error()}")
        print("\nMake sure:")
        print("  1. MetaTrader 5 terminal is installed")
        print("  2. MT5 is running and logged in to your broker")
        print("  3. You're running this on Windows")
        sys.exit(1)

    # Get account info
    account_info = mt5.account_info()
    if account_info:
        print(f"Connected to: {account_info.server}")
        print(f"Account:      {account_info.login}")
        print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Download each symbol
    success_count = 0
    for symbol in args.symbols:
        print(f"Downloading {symbol}...")

        # Check if symbol exists
        symbol_info = mt5.symbol_info(symbol)
        if symbol_info is None:
            print(f"  ERROR: Symbol {symbol} not found. Check broker symbol names.")
            # Try common variations
            variations = [
                symbol,
                f"{symbol}.a",  # Some brokers add suffix
                f"{symbol}.i",
                f"{symbol}m",
                f"{symbol}micro",
            ]
            print(f"  Try one of these if available in your broker:")
            for var in variations:
                info = mt5.symbol_info(var)
                if info:
                    print(f"    - {var}")
            continue

        # Make sure symbol is visible in Market Watch
        if not symbol_info.visible:
            mt5.symbol_select(symbol, True)

        # Download data
        df = download_symbol_data(symbol, args.timeframe, start_date, end_date)

        if df is not None and len(df) > 0:
            # Save to parquet
            filename = f"{symbol}_{args.timeframe}.parquet"
            filepath = output_dir / filename
            df.to_parquet(filepath)

            print(f"  Saved {len(df):,} bars to {filepath}")
            print(f"  Date range: {df.index.min()} to {df.index.max()}")
            success_count += 1
        else:
            print(f"  FAILED: No data downloaded for {symbol}")

        print()

    # Cleanup
    mt5.shutdown()

    print("=" * 50)
    print(f"Completed: {success_count}/{len(args.symbols)} symbols downloaded")

    if success_count > 0:
        print(f"\nTo run backtest with this data:")
        print(f"  .venv/bin/fx-trading backtest configs/your_config.yaml")


if __name__ == "__main__":
    main()