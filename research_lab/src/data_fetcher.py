"""
Market data fetching via ccxt with pagination support.
"""

import logging
from datetime import datetime, timezone

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)


def fetch_ohlcv(
    symbol: str = "BTC/USDT",
    timeframe: str = "1h",
    since_unix_ms: int | None = None,
    limit: int = 1000,
    max_candles: int = 35040,  # ~4 years of hourly data
) -> pd.DataFrame:
    """
    Download historical OHLCV data from Binance.

    Handles pagination by looping through time ranges.
    Returns a DataFrame indexed by UTC timestamp.
    """
    exchange = ccxt.binance({"enableRateLimit": True})

    if since_unix_ms is None:
        four_years_ago = datetime(2022, 1, 1, tzinfo=timezone.utc)
        since_unix_ms = int(four_years_ago.timestamp() * 1000)

    all_candles: list = []
    current_since = since_unix_ms

    logger.info(
        "Fetching %s %s data from %s...",
        symbol,
        timeframe,
        datetime.fromtimestamp(current_since / 1000, tz=timezone.utc),
    )

    while len(all_candles) < max_candles:
        try:
            ohlcv = exchange.fetch_ohlcv(
                symbol, timeframe, since=current_since, limit=limit
            )
        except ccxt.NetworkError as e:
            logger.error("Network error fetching OHLCV: %s", e)
            break
        except ccxt.ExchangeError as e:
            logger.error("Exchange error: %s", e)
            break

        if not ohlcv:
            break

        all_candles.extend(ohlcv)

        last_ts = ohlcv[-1][0]
        current_since = last_ts + 1

        logger.info(
            "  Fetched %d candles (up to %s)",
            len(all_candles),
            datetime.fromtimestamp(last_ts / 1000, tz=timezone.utc),
        )

        if len(ohlcv) < limit:
            break

    if not all_candles:
        raise ValueError(f"No data returned for {symbol} {timeframe}")

    df = pd.DataFrame(
        all_candles,
        columns=["timestamp", "open", "high", "low", "close", "volume"],
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df = df.set_index("timestamp")
    df = df.iloc[:max_candles]

    logger.info("Total candles fetched: %d", len(df))
    return df


def validate_data(df: pd.DataFrame, symbol: str, timeframe: str = "1h") -> None:
    """Check for common data quality issues."""
    issues: list[str] = []

    freq_minutes = {"1h": 60, "15m": 15, "5m": 5, "1d": 1440}
    freq = freq_minutes.get(timeframe, 60)
    expected_delta = pd.Timedelta(minutes=freq)
    actual_deltas = df.index.to_series().diff().dropna()
    gaps = actual_deltas[actual_deltas > expected_delta * 1.5]

    if len(gaps) > 0:
        issues.append(f"Found {len(gaps)} gaps in data (missing candles)")

    if (df["close"] <= 0).any():
        issues.append("Found zero or negative close prices")
    if (df["volume"] < 0).any():
        issues.append("Found negative volume")

    returns = df["close"].pct_change().abs()
    outliers = returns[returns > 0.5]
    if len(outliers) > 0:
        issues.append(
            f"Found {len(outliers)} candles with >50% single-candle price change"
        )

    if issues:
        logger.warning("Data quality issues for %s:", symbol)
        for issue in issues:
            logger.warning("  - %s", issue)
    else:
        logger.info(
            "Data validation passed for %s (%d candles, no issues)",
            symbol,
            len(df),
        )


