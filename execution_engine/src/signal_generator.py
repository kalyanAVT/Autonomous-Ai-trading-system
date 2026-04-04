"""Model inference pipeline for live trading signals.

Loads a trained PPO model from research_lab and produces trading signals
from real-time feature snapshots.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

from .models import MarketSnapshot

logger = logging.getLogger(__name__)

# Feature column order matching feature_engine.compute_all() in research_lab.
# DO NOT change this order without retraining the model.
FEATURE_COLUMNS = [
    "log_ret_1",
    "log_ret_3",
    "log_ret_12",
    "log_ret_24",
    "roll_vol_12",
    "roll_vol_24",
    "roll_mean_12",
    "roll_mean_24",
    "momentum_14",
    "vol_change_12",
    "hl_spread",
    "price_pos_24",
]


class Signal:
    """Normalized trading signal [-1.0, 1.0] from the model."""

    def __init__(self, action: float, confidence: float, snapshot: MarketSnapshot):
        self.action = np.clip(action, -1.0, 1.0)
        self.confidence = confidence
        self.timestamp = snapshot.timestamp
        self.price = snapshot.close

    @property
    def is_long(self) -> bool:
        return self.action > 0.05

    @property
    def is_short(self) -> bool:
        return self.action < -0.05

    @property
    def is_flat(self) -> bool:
        return abs(self.action) <= 0.05

    def __repr__(self) -> str:
        direction = "LONG" if self.is_long else ("SHORT" if self.is_short else "FLAT")
        return f"Signal({direction}, action={self.action:.4f}, price={self.price:.2f})"


class SignalGenerator:
    """Wraps a trained PPO model to produce actionable trading signals."""

    def __init__(self, model_path: str):
        path = Path(model_path)
        if not path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")

        logger.info("Loading PPO model from %s", path)
        self.model = PPO.load(str(path), device="cpu")
        logger.info("Model loaded successfully")

    def predict(self, snapshot: MarketSnapshot) -> Signal:
        """Run model inference on a feature snapshot.

        Args:
            snapshot: MarketSnapshot with feature dict computed by DataFeed.

        Returns:
            Signal object with action value [-1, 1] and metadata.
        """
        if snapshot.features is None:
            raise ValueError("Snapshot has no features — need more historical data")

        # Build observation vector using the same column order the model was trained on
        feature_vec = np.array(
            [snapshot.features[name] for name in FEATURE_COLUMNS],
            dtype=np.float32,
        ).reshape(1, -1)

        # Clip to match training-time normalization
        feature_vec = np.clip(feature_vec, -10.0, 10.0)

        action, _states = self.model.predict(
            feature_vec,
            deterministic=True,
        )

        # Action confidence from raw action magnitude
        confidence = float(abs(action[0]))

        return Signal(
            action=float(action[0]),
            confidence=confidence,
            snapshot=snapshot,
        )
