"""Real-time market data feed using ccxt.

Polls OHLCV from the exchange and computes features using the same
pipeline as research_lab to ensure train/serve parity.
"""

from __future__ import annotations

import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import ccxt
import numpy as np
import pandas as pd

from .config import Settings
from .models import OHLCVCandle, MarketSnapshot

logger = logging.getLogger(__name__)

# Add research_lab to path for feature_engine import
_research_src = Path(__file__).resolve().parent.parent.parent / "research_lab" / "src"
if str(_research_src) not in sys.path:
    sys.path.insert(0, str(_research_src))

from feature_engine import FeatureEngine


class DataFeed:
    """Polls OHLCV data and produces aligned feature vectors."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.exchange = ccxt.binance({
            "enableRateLimit": True,
            "options": {"defaultType": "spot"},
        })
        self._candle_buffer: list[OHLCVCandle] = []

    async def fetch_history(self, limit: Optional[int] = None) -> list[OHLCVCandle]:
        """Fetch historical candles to seed the feature engine.

        Returns at least ``feature_lookback`` candles plus a safety margin.
        """
        count = limit or self.settings.feature_lookback + 100
        logger.info("Fetching %d historical candles for %s", count, self.settings.symbol)

        try:
            raw = self.exchange.fetch_ohlcv(
                self.settings.symbol,
                self.settings.timeframe,
                limit=count,
            )
        except ccxt.NetworkError as e:
            logger.error("Network error fetching OHLCV: %s", e)
            return []
        except ccxt.ExchangeError as e:
            logger.error("Exchange error: %s", e)
            return []

        candles = [
            OHLCVCandle(
                timestamp=datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc),
                open=float(c[1]),
                high=float(c[2]),
                low=float(c[3]),
                close=float(c[4]),
                volume=float(c[5]),
            )
            for c in raw
        ]
        self._candle_buffer = candles
        logger.info("Fetched %d candles", len(candles))
        return candles

    def compute_features(self, candles: Optional[list[OHLCVCandle]] = None) -> MarketSnapshot:
        """Compute feature vector from candle history.

        Uses the same FeatureEngine from research_lab to ensure
        consistency between training and live inference.
        """
        candle_list = candles or self._candle_buffer
        if not candle_list or len(candle_list) < 550:
            raise ValueError(
                f"Need >= 550 candles for features, got {len(candle_list) if candle_list else 0}"
            )

        df = pd.DataFrame(
            {
                "open": [c.open for c in candle_list],
                "high": [c.high for c in candle_list],
                "low": [c.low for c in candle_list],
                "close": [c.close for c in candle_list],
                "volume": [c.volume for c in candle_list],
            }
        )
        df.index = pd.DatetimeIndex([c.timestamp for c in candle_list], tz="UTC")

        engine = FeatureEngine(df)
        features_df, close_prices = engine.compute_all()

        # Take the most recent feature row
        last_features = features_df.iloc[-1]
        last_close = float(close_prices[-1])

        feature_dict = {col: float(last_features[col]) for col in features_df.columns}

        return MarketSnapshot(
            timestamp=candle_list[-1].timestamp,
            symbol=self.settings.symbol,
            close=last_close,
            features=feature_dict,
        )

    async def refresh(
        self,
        snapshot: Optional[MarketSnapshot],
        n_candles: int = 5,
    ) -> MarketSnapshot:
        """Poll for the latest candles and compute an updated feature snapshot.

        Fetches ``n_candles`` recent candles, appends to the buffer,
        removes duplicates, and re-computes features.
        """
        count = max(n_candles, 50)
        try:
            raw = self.exchange.fetch_ohlcv(
                self.settings.symbol,
                self.settings.timeframe,
                limit=count,
            )
        except ccxt.NetworkError as e:
            logger.error("Network error on refresh: %s", e)
            if snapshot is None:
                raise
            return snapshot
        except ccxt.ExchangeError as e:
            logger.error("Exchange error on refresh: %s", e)
            if snapshot is None:
                raise
            return snapshot

        new_candles = [
            OHLCVCandle(
                timestamp=datetime.fromtimestamp(int(c[0]) / 1000, tz=timezone.utc),
                open=float(c[1]),
                high=float(c[2]),
                low=float(c[3]),
                close=float(c[4]),
                volume=float(c[5]),
            )
            for c in raw
        ]

        # Merge with buffer (deduplicate by timestamp)
        seen = {c.timestamp for c in self._candle_buffer}
        for c in new_candles:
            if c.timestamp not in seen:
                self._candle_buffer.append(c)
                seen.add(c.timestamp)

        # Keep only the most recent N candles to avoid memory bloat
        max_buffer = self.settings.feature_lookback + 200
        if len(self._candle_buffer) > max_buffer:
            self._candle_buffer = self._candle_buffer[-max_buffer:]

        return self.compute_features(self._candle_buffer)
