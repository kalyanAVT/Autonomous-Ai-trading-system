"""
Main entry point for the research data pipeline.
Fetches OHLCV from Binance, validates, computes features, saves as parquet.

Run from the src/ directory:
    python data_preprocessor.py --symbol BTC/USDT
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as standalone script from src/ directory
_parent = Path(__file__).resolve().parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from data_fetcher import fetch_ohlcv, validate_data
from feature_engine import FeatureEngine

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch and prepare market data for RL training"
    )
    parser.add_argument("--symbol", default="BTC/USDT", help="Trading pair")
    parser.add_argument("--timeframe", default="1h", help="Candle timeframe")
    parser.add_argument(
        "--output", default="../data/market_data.parquet", help="Output parquet path"
    )
    parser.add_argument(
        "--no-features",
        action="store_true",
        help="Skip technical feature computation",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    raw_df = fetch_ohlcv(symbol=args.symbol, timeframe=args.timeframe)
    validate_data(raw_df, symbol=args.symbol, timeframe=args.timeframe)

    if not args.no_features:
        engine = FeatureEngine(raw_df)
        features, close_prices = engine.compute_all()
        logger.info(
            "Features computed: %d columns, %d rows",
            features.shape[1],
            features.shape[0],
        )

        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        features.to_parquet(str(output_path))
        logger.info(
            "Saved features to %s (%.1f KB)",
            output_path,
            output_path.stat().st_size / 1024,
        )

        # Save aligned close prices as numpy array
        prices_path = output_path.with_suffix(".npy")
        np.save(str(prices_path), close_prices)
        logger.info(
            "Saved close prices to %s (%d values)",
            prices_path,
            len(close_prices),
        )
    else:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        raw_df.to_parquet(str(output_path))
        logger.info("Saved raw OHLCV to %s", output_path)


if __name__ == "__main__":
    main()
