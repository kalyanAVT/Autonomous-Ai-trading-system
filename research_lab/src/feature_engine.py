"""
Feature engineering for the trading environment.
Takes raw OHLCV and produces normalized features for RL input.
"""

import pandas as pd
import numpy as np
from typing import Optional


class FeatureEngine:
    """
    Computes technical indicators and normalizes features for RL input.

    Features produced:
    - Log returns (1h, 3h, 12h, 24h)
    - Rolling volatility (12h, 24h windows)
    - Rolling mean close price (12h, 24h windows)
    - Momentum (RSI-like, 14h window)
    - Volume changes (rolling 12h mean volume change)
    - High-Low spread (volatility proxy)
    - Price position within recent range (where is price relative to 24h high/low)

    All features are z-scored over a rolling window to keep values bounded.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.copy()
        self._feature_cols: list[str] = []

    def _add_log_returns(self, window: int, label: Optional[str] = None) -> str:
        """Log return over N periods."""
        name = f"log_ret_{window}" if label is None else label
        self.df[name] = np.log(self.df["close"] / self.df["close"].shift(window))
        self._feature_cols.append(name)
        return name

    def _add_rolling_vol(self, window: int, src_col: str = "close") -> str:
        """Rolling standard deviation of returns."""
        name = f"roll_vol_{window}"
        returns = self.df[src_col].pct_change()
        self.df[name] = returns.rolling(window=window).std()
        self._feature_cols.append(name)
        return name

    def _add_rolling_mean(self, window: int, src_col: str = "close") -> str:
        """Rolling mean of a column."""
        name = f"roll_mean_{window}"
        self.df[name] = self.df[src_col].rolling(window=window).mean()
        self._feature_cols.append(name)
        return name

    def _add_momentum_rsi(self, window: int = 14) -> str:
        """RSI-like momentum indicator."""
        name = f"momentum_{window}"
        delta = self.df["close"].diff()
        gain = delta.clip(lower=0)
        loss = (-delta).clip(lower=0)
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        rs = avg_gain / (avg_loss + 1e-10)
        self.df[name] = 100.0 - (100.0 / (1.0 + rs))
        self._feature_cols.append(name)
        return name

    def _add_volume_change(self, window: int = 12) -> str:
        """Rolling mean of volume change percentage."""
        name = f"vol_change_{window}"
        vol_pct_change = self.df["volume"].pct_change()
        self.df[name] = vol_pct_change.rolling(window=window).mean()
        self._feature_cols.append(name)
        return name

    def _add_hl_spread(self) -> str:
        """High-Low spread as volatility proxy."""
        name = "hl_spread"
        self.df[name] = (self.df["high"] - self.df["low"]) / (self.df["close"] + 1e-10)
        self._feature_cols.append(name)
        return name

    def _add_price_position(self, window: int = 24) -> str:
        """Where is current price relative to recent high/low range? (0-1)"""
        name = f"price_pos_{window}"
        rolling_high = self.df["high"].rolling(window=window).max()
        rolling_low = self.df["low"].rolling(window=window).min()
        range_size = rolling_high - rolling_low
        self.df[name] = (self.df["close"] - rolling_low) / (range_size + 1e-10)
        self._feature_cols.append(name)
        return name

    def compute_all(self) -> tuple[pd.DataFrame, np.ndarray]:
        """Compute all features and return the feature DataFrame and close prices.

        Returns:
            A tuple of (features DataFrame, close_prices array) where close_prices
            is aligned with the rows returned in features.

        Raises:
            ValueError: If input DataFrame has fewer than 550 rows.
        """
        if len(self.df) < 550:
            raise ValueError(
                f"Need at least 550 rows for feature computation, got {len(self.df)}"
            )

        # Returns at multiple timeframes
        self._add_log_returns(1)
        self._add_log_returns(3)
        self._add_log_returns(12)
        self._add_log_returns(24)

        # Rolling volatility
        self._add_rolling_vol(12)
        self._add_rolling_vol(24)

        # Rolling mean close (relative to current price later)
        self._add_rolling_mean(12)
        self._add_rolling_mean(24)

        # Momentum
        self._add_momentum_rsi(14)

        # Volume dynamics
        self._add_volume_change(12)

        # Volatility proxies
        self._add_hl_spread()
        self._add_price_position(24)

        # Build feature subset (drop raw OHLCV after rolling features computed)
        features = self.df[self._feature_cols].copy()

        # Z-score normalize each feature with expanding window to avoid look-ahead bias
        features = self._rolling_zscore(features, window=500)

        # Drop NaN rows at the start (from rolling calculations)
        # Keep track of the resulting index to align close prices
        features = features.iloc[50:].dropna()

        # Extract aligned close prices — same index as the final features
        close_prices = self.df["close"].loc[features.index].values.astype(np.float64)

        return features, close_prices

    def feature_columns(self) -> list[str]:
        """Return the ordered list of feature column names (single source of truth).

        Import this from FeatureEngine to avoid hardcoding column order
        in signal_generator.py or anywhere else.
        """
        if not self._feature_cols:
            # Compute features to populate _feature_cols
            self.compute_all()
        return list(self._feature_cols)

    @staticmethod
    def _rolling_zscore(df: pd.DataFrame, window: int) -> pd.DataFrame:
        """Z-score normalize each column with an expanding mean/std window."""
        result = df.copy()
        for col in df.columns:
            rolling_mean = result[col].rolling(window=window, min_periods=50).mean()
            rolling_std = result[col].rolling(window=window, min_periods=50).std()
            result[col] = (result[col] - rolling_mean) / (rolling_std + 1e-10)
        return result
