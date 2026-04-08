"""Standalone data downloader — Phase 1 data plumbing.

Downloads historical OHLCV from ccxt, saves as parquet,
computes features via FeatureEngine, and outputs aligned
features + close prices for backtesting/training.

Usage:
    python download_data.py --symbol BTC/USDT --timeframe 1h --limit 35040 --output data/
    # 35040 = 4 years of hourly candles

Outputs:
    data/<symbol>_<timeframe>.parquet      — raw OHLCV
    data/<symbol>_<timeframe>_features.parquet — normalized features
    data/<symbol>_<timeframe>_prices.npy   — aligned close prices
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import ccxt
import numpy as np
import pandas as pd


def download_ohlcv(
    symbol: str,
    timeframe: str,
    limit: int,
    exchange: str = "binance",
    progress: bool = True,
) -> pd.DataFrame:
    """Fetch OHLCV candles from the exchange.

    Uses pagination because ccxt limits each call to ~1000 candles.
    """
    exchange_client = getattr(ccxt, exchange)({
        "enableRateLimit": True,
        "options": {"defaultType": "spot"},
    })

    columns = ["timestamp", "open", "high", "low", "close", "volume"]
    all_candles: list[list] = []

    since = None
    while len(all_candles) < limit:
        batch = min(1000, limit - len(all_candles))
        kwargs: dict = {"limit": batch}
        if since is not None:
            kwargs["since"] = since

        raw = exchange_client.fetch_ohlcv(symbol, timeframe, **kwargs)
        if not raw:
            break

        all_candles.extend(raw)
        if progress:
            print(f"  Downloaded {len(all_candles)}/{limit} candles")

        if len(raw) < batch:
            break

        since = raw[-1][0] + 1

    if not all_candles:
        raise RuntimeError(f"No data fetched for {symbol} on {timeframe}")

    df = pd.DataFrame(all_candles, columns=columns)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df.set_index("timestamp", inplace=True)
    df.sort_index(inplace=True)

    # Remove duplicates
    df = df[~df.index.duplicated(keep="first")]

    return df


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download OHLCV data and compute features for training/backtesting"
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument(
        "--limit", type=int, default=35040,
        help="Number of candles (35040 = 4 years of 1h data)"
    )
    parser.add_argument("--exchange", default="binance")
    parser.add_argument(
        "--output", default="data",
        help="Output directory for files"
    )
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    name = f"{args.symbol.replace('/', '_')}_{args.timeframe}"

    # Step 1: Download raw OHLCV
    print(f"\nDownloading {args.limit} {args.timeframe} candles for {args.symbol}...")
    df = download_ohlcv(args.symbol, args.timeframe, args.limit, args.exchange)
    print(f"Downloaded {len(df)} candles: {df.index[0]} to {df.index[-1]}")

    raw_path = output_dir / f"{name}.parquet"
    df.to_parquet(raw_path)
    print(f"Raw OHLCV saved to {raw_path}")

    # Step 2: Compute features
    print(f"\nComputing features (need >= 550 candles, got {len(df)})...")
    if len(df) < 550:
        print("ERROR: Need at least 550 candles for feature computation.")
        sys.exit(1)

    _research_src = Path(__file__).resolve().parent.parent / "research_lab" / "src"
    if str(_research_src) not in sys.path:
        sys.path.insert(0, str(_research_src))

    from feature_engine import FeatureEngine  # noqa: E402

    engine = FeatureEngine(df)
    features_df, close_prices = engine.compute_all()

    features_path = output_dir / f"{name}_features.parquet"
    features_df.to_parquet(features_path)
    print(f"Features saved to {features_path} ({len(features_df)} rows, {len(features_df.columns)} cols)")

    prices_path = output_dir / f"{name}_prices.npy"
    np.save(prices_path, close_prices)
    print(f"Close prices saved to {prices_path} ({len(close_prices)} values)")

    print(f"\n{'='*50}")
    print("  Data download complete")
    print(f"  Symbol:       {args.symbol}")
    print(f"  Timeframe:    {args.timeframe}")
    print(f"  Candles:      {len(df)}")
    print(f"  Features:     {len(features_df)}")
    print(f"  Feature cols: {list(features_df.columns)}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    main()
