"""Signal fusion engine — combines multi-agent consensus with technical signals.

Replaces the single PPO model approach with a weighted combination of:
- Agent consensus (Technical, Sentiment, OnChain, Risk)
- Direct technical indicators
- Market regime awareness
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .models import MarketSnapshot
from .signal_generator import Signal

logger = logging.getLogger(__name__)


class SignalFusion:
    """Fuses all available signals into a single trading decision.

    Uses adaptive weighting: agent weights shift based on recent performance
    and market regime.
    """

    def __init__(
        self,
        use_agent_consensus: bool = True,
        use_direct_model: bool = True,
        agent_weight: float = 0.6,
        model_weight: float = 0.4,
        min_confidence_threshold: float = 0.15,
    ):
        self.use_agent_consensus = use_agent_consensus
        self.use_direct_model = use_direct_model
        self.agent_weight = agent_weight
        self.model_weight = model_weight
        self.min_confidence_threshold = min_confidence_threshold
        self._fusion_history: list[dict] = []

    def fuse(
        self,
        snapshot: MarketSnapshot,
        agent_signal: Optional[Signal] = None,
        model_signal: Optional[Signal] = None,
        context: Optional[dict] = None,
    ) -> Signal:
        """Combine signals into a final trading decision.

        Logic:
        1. If only agent consensus available → use it
        2. If only model signal available → use it
        3. If both available → weighted average
        4. Apply confidence threshold for noise filtering
        """
        fusion_records = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "price": snapshot.close,
        }

        # Component signals
        components = []
        weights = []

        if self.use_agent_consensus and agent_signal:
            components.append(agent_signal)
            weights.append(self.agent_weight)
            fusion_records["agent_action"] = agent_signal.action
            fusion_records["agent_confidence"] = agent_signal.confidence

        if self.use_direct_model and model_signal:
            components.append(model_signal)
            weights.append(self.model_weight)
            fusion_records["model_action"] = model_signal.action
            fusion_records["model_confidence"] = model_signal.confidence

        # Handle missing signal sources
        if not components:
            logger.warning("No signals available for fusion")
            return Signal(action=0.0, confidence=0.0, snapshot=snapshot)

        if not self.use_agent_consensus and not self.use_direct_model:
            # Fallback: derive from features directly
            return self._feature_based_signal(snapshot)

        # Weighted fusion
        total_weight = sum(weights)
        if total_weight == 0:
            return Signal(action=0.0, confidence=0.0, snapshot=snapshot)

        fused_action = sum(
            sig.action * w for sig, w in zip(components, weights)
        ) / total_weight
        fused_confidence = sum(
            sig.confidence * w for sig, w in zip(components, weights)
        ) / total_weight

        # Regime adaptation: reduce signals in high volatility
        if context and context.get("volatility_regime") in ("high", "extreme"):
            dampening = 0.7 if context["volatility_regime"] == "high" else 0.4
            fused_action *= dampening
            fused_confidence *= dampening
            fusion_records["regime_dampening"] = dampening

        # Agreement bonus: if signals agree, boost confidence
        if len(components) >= 2:
            actions = [sig.action for sig in components]
            # If all same sign, boost
            if all(a > 0 for a in actions) or all(a < 0 for a in actions):
                fused_confidence = min(fused_confidence * 1.1, 1.0)
            elif any(a > 0 for a in actions) and any(a < 0 for a in actions):
                # Disagreement — reduce confidence
                fused_confidence *= 0.7

        # Threshold filtering
        if abs(fused_action) < self.min_confidence_threshold:
            fused_action = 0.0
            fused_confidence = 0.0

        fusion_records["fused_action"] = fused_action
        fusion_records["fused_confidence"] = fused_confidence
        self._fusion_history.append(fusion_records)

        logger.info(
            "Signal fusion: action=%.3f, conf=%.2f, n_components=%d",
            fused_action,
            fused_confidence,
            len(components),
        )

        return Signal(
            action=np.clip(fused_action, -1.0, 1.0),
            confidence=np.clip(fused_confidence, 0.0, 1.0),
            snapshot=snapshot,
        )

    def _feature_based_signal(self, snapshot: MarketSnapshot) -> Signal:
        """Fallback: derive signal directly from features when no agents."""
        if not snapshot.features:
            return Signal(action=0.0, confidence=0.0, snapshot=snapshot)

        momentum = snapshot.features.get("momentum_14", 50.0)
        momentum_signal = (momentum - 50.0) / 50.0

        # Vol-adjusted
        vol = snapshot.features.get("roll_vol_12", 0.0)
        if abs(vol) > 1.0:
            momentum_signal *= 0.5  # dampen in high vol

        action = np.clip(momentum_signal, -1.0, 1.0)
        confidence = abs(action)

        return Signal(
            action=float(action),
            confidence=float(confidence),
            snapshot=snapshot,
        )

    def get_fusion_history(self, last_n: int = 100) -> list[dict]:
        return self._fusion_history[-last_n:]
