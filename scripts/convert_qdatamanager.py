#!/usr/bin/env python3
"""
Convert QuantDataManager CSV exports to parquet format for backtesting.

QuantDataManager typically exports in format:
    timestamp,open,high,low,close,volume
or:
    Date,Time,Open,High,Low,Close,Volume

Usage:
    python scripts/convert_qdatamanager.py data/raw/EURUSD_M5.csv --symbol EURUSD
    python scripts/convert_qdatamanager.py data/raw/*.csv --auto-symbol
    python scripts/convert_qdatamanager.py data/raw/ --output data/sample
"""

import argparse
import sys
from pathlib import Path

import pandas as pd


def detect_format(df: pd.DataFrame) -> str:
    """Detect the CSV format based on columns."""
    cols_lower = [c.lower() for c in df.columns]

    if "timestamp" in cols_lower or "datetime" in cols_lower:
        return "timestamp"
    elif "date" in cols_lower and "time" in cols_lower:
        return "date_time"
    elif "time" in cols_lower and "open" in cols_lower:
        return "time_ohlc"
    else:
        # Check if first column looks like a datetime
        first_col = df.columns[0]
        try:
            pd.to_datetime(df[first_col].iloc[0])
            return "first_col_datetime"
        except:
            pass

    return "unknown"


def convert_csv_to_parquet(
    csv_path: Path,
    symbol: str,
    output_dir: Path,
    timeframe: str = "M5",
) -> Path | None:
    """Convert a single CSV file to parquet format."""

    print(f"Converting {csv_path.name}...")

    # Read CSV
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        print(f"  ERROR reading CSV: {e}")
        return None

    if len(df) == 0:
        print(f"  ERROR: Empty file")
        return None

    print(f"  Found {len(df):,} rows, columns: {list(df.columns)}")

    # Detect format and parse datetime
    fmt = detect_format(df)
    print(f"  Detected format: {fmt}")

    try:
        if fmt == "timestamp":
            # Column named 'timestamp' or 'Timestamp' or 'datetime'
            time_col = [c for c in df.columns if c.lower() in ("timestamp", "datetime")][0]
            df["Datetime"] = pd.to_datetime(df[time_col])

        elif fmt == "date_time":
            # Separate Date and Time columns
            date_col = [c for c in df.columns if c.lower() == "date"][0]
            time_col = [c for c in df.columns if c.lower() == "time"][0]
            df["Datetime"] = pd.to_datetime(df[date_col] + " " + df[time_col])

        elif fmt == "time_ohlc":
            # First column might be unnamed datetime
            time_col = [c for c in df.columns if c.lower() == "time"][0]
            df["Datetime"] = pd.to_datetime(df[time_col])

        elif fmt == "first_col_datetime":
            # First column is datetime (possibly unnamed)
            df["Datetime"] = pd.to_datetime(df.iloc[:, 0])

        else:
            # Try to parse first column as datetime
            print(f"  Attempting to parse first column as datetime...")
            df["Datetime"] = pd.to_datetime(df.iloc[:, 0])

    except Exception as e:
        print(f"  ERROR parsing datetime: {e}")
        print(f"  First few rows:\n{df.head()}")
        return None

    # Normalize column names
    col_mapping = {}
    for col in df.columns:
        col_lower = col.lower()
        if col_lower in ("open", "o"):
            col_mapping[col] = "open"
        elif col_lower in ("high", "h"):
            col_mapping[col] = "high"
        elif col_lower in ("low", "l"):
            col_mapping[col] = "low"
        elif col_lower in ("close", "c"):
            col_mapping[col] = "close"
        elif col_lower in ("volume", "vol", "v", "tickvol", "tick_volume"):
            col_mapping[col] = "volume"

    df = df.rename(columns=col_mapping)

    # Ensure we have required columns
    required = ["open", "high", "low", "close"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"  ERROR: Missing required columns: {missing}")
        print(f"  Available columns: {list(df.columns)}")
        return None

    # Add volume if missing
    if "volume" not in df.columns:
        df["volume"] = 0

    # Add symbol
    df["symbol"] = symbol

    # Set datetime as index with UTC timezone
    df["Datetime"] = pd.to_datetime(df["Datetime"], utc=True)
    df = df.set_index("Datetime")
    df = df.sort_index()

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    # Select final columns in correct order
    df = df[["open", "high", "low", "close", "volume", "symbol"]]

    # Save to parquet
    output_path = output_dir / f"{symbol}_{timeframe}.parquet"
    df.to_parquet(output_path)

    print(f"  Saved {len(df):,} bars to {output_path}")
    print(f"  Date range: {df.index.min()} to {df.index.max()}")

    return output_path


def extract_symbol_from_filename(filename: str) -> str | None:
    """Try to extract symbol from filename."""
    # Common patterns: EURUSD_M5.csv, EURUSD-M5.csv, EURUSD.csv
    name = Path(filename).stem.upper()

    symbols = ["EURUSD", "GBPUSD", "USDJPY", "XAUUSD", "AUDUSD", "USDCAD", "NZDUSD", "USDCHF"]

    for sym in symbols:
        if sym in name:
            return sym

    # If name looks like a symbol (6 chars, all letters)
    if len(name) == 6 and name.isalpha():
        return name

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Convert QuantDataManager CSV to parquet",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Convert single file with explicit symbol
    python scripts/convert_qdatamanager.py data/raw/EURUSD_M5.csv --symbol EURUSD

    # Convert all CSVs in directory, auto-detect symbols from filenames
    python scripts/convert_qdatamanager.py data/raw/*.csv --auto-symbol

    # Convert directory to specific output
    python scripts/convert_qdatamanager.py data/raw/ --output data/sample --auto-symbol
        """
    )

    parser.add_argument(
        "input",
        nargs="+",
        help="CSV file(s) or directory to convert",
    )
    parser.add_argument(
        "--symbol",
        help="Symbol name (required for single file unless --auto-symbol)",
    )
    parser.add_argument(
        "--auto-symbol",
        action="store_true",
        help="Auto-detect symbol from filename",
    )
    parser.add_argument(
        "--timeframe",
        default="M5",
        help="Timeframe label for output filename (default: M5)",
    )
    parser.add_argument(
        "--output",
        default="data/sample",
        help="Output directory (default: data/sample)",
    )

    args = parser.parse_args()

    # Collect input files
    input_files = []
    for path_str in args.input:
        path = Path(path_str)
        if path.is_dir():
            input_files.extend(path.glob("*.csv"))
        elif path.exists():
            input_files.append(path)
        elif "*" in path_str:
            # Glob pattern already expanded by shell
            input_files.append(path)

    if not input_files:
        print("ERROR: No CSV files found")
        sys.exit(1)

    print(f"Found {len(input_files)} file(s) to convert")
    print()

    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Convert each file
    success = 0
    for csv_path in input_files:
        # Determine symbol
        if args.symbol:
            symbol = args.symbol.upper()
        elif args.auto_symbol:
            symbol = extract_symbol_from_filename(csv_path.name)
            if not symbol:
                print(f"Skipping {csv_path.name}: Could not detect symbol")
                continue
        else:
            print(f"ERROR: Specify --symbol or use --auto-symbol")
            sys.exit(1)

        result = convert_csv_to_parquet(csv_path, symbol, output_dir, args.timeframe)
        if result:
            success += 1
        print()

    print(f"Converted {success}/{len(input_files)} files")

    if success > 0:
        print(f"\nTo run backtest:")
        print(f"  .venv/bin/fx-trading backtest configs/backtest_optimized.yaml")


if __name__ == "__main__":
    main()