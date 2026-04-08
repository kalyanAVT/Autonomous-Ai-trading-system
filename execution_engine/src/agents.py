"""Multi-agent decision framework for autonomous trading.

Each agent provides an independent trading signal based on different data sources.
A consensus engine combines them with learned weights into a final signal.
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional

import numpy as np

from .models import MarketSnapshot
from .signal_generator import Signal

logger = logging.getLogger(__name__)


@dataclass
class AgentSignal:
    """Output from a single agent."""

    name: str
    action: float  # -1.0 (sell) to 1.0 (buy)
    confidence: float  # 0.0 to 1.0
    timestamp: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    metadata: dict = field(default_factory=dict)

    def __repr__(self) -> str:
        direction = "BUY" if self.action > 0.05 else ("SELL" if self.action < -0.05 else "HOLD")
        return f"{self.name}({direction}, action={self.action:.3f}, conf={self.confidence:.2f})"


class BaseAgent(ABC):
    """Base class for all trading agents."""

    name: str = "base"

    def __init__(self, weight: float = 1.0):
        self.weight = weight
        self.total_decisions = 0
        self.correct_decisions = 0
        self._last_signal: Optional[AgentSignal] = None

    @abstractmethod
    def produce_signal(self, snapshot: MarketSnapshot, context: dict) -> AgentSignal:
        """Produce a trading signal based on this agent's specialty."""
        ...

    def update_performance(self, correct: bool) -> None:
        """Track agent accuracy for dynamic weight adjustment."""
        self.total_decisions += 1
        if correct:
            self.correct_decisions += 1

    @property
    def accuracy(self) -> float:
        if self.total_decisions == 0:
            return 0.5
        return self.correct_decisions / self.total_decisions

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(weight={self.weight}, acc={self.accuracy:.2f})"


class TechnicalAgent(BaseAgent):
    """Agent using technical indicators and ML model signals."""

    name = "technical"

    def __init__(self, weight: float = 1.0, model_signal: Optional[Signal] = None):
        super().__init__(weight=weight)
        self.model_signal = model_signal

    def produce_signal(self, snapshot: MarketSnapshot, context: dict) -> AgentSignal:
        if self.model_signal:
            action = float(self.model_signal.action)
            confidence = float(self.model_signal.confidence)
        else:
            # Fallback: simple momentum from features
            if snapshot.features:
                momentum = snapshot.features.get("momentum_14", 50.0)
                action = (momentum - 50.0) / 50.0
                confidence = abs(action)
            else:
                action = 0.0
                confidence = 0.0

        signal = AgentSignal(
            name=self.name,
            action=np.clip(action, -1.0, 1.0),
            confidence=np.clip(confidence, 0.0, 1.0),
            metadata={"price": snapshot.close, "symbol": snapshot.symbol},
        )
        self._last_signal = signal
        return signal


class SentimentAgent(BaseAgent):
    """Agent using social sentiment from Reddit, news, and Google Trends."""

    name = "sentiment"

    def produce_signal(self, snapshot: MarketSnapshot, context: dict) -> AgentSignal:
        reddit_score = context.get("reddit_sentiment", 0.0)
        news_score = context.get("news_sentiment", 0.0)
        trends_score = context.get("google_trends", 0.0)
        fear_greed = context.get("fear_greed", 50)

        # Convert fear/greed to contrarian signal
        # Extreme fear (low) = buy opportunity, extreme greed (high) = sell risk
        fg_signal = (50.0 - fear_greed) / 50.0

        # Weighted sentiment composite
        weights = {
            "reddit": 0.3,
            "news": 0.3,
            "trends": 0.15,
            "fear_greed": 0.25,
        }

        action = (
            weights["reddit"] * reddit_score
            + weights["news"] * news_score
            + weights["trends"] * trends_score
            + weights["fear_greed"] * fg_signal
        )

        # Confidence from agreement between sources
        scores = [reddit_score, news_score, trends_score, fg_signal]
        agreement = 1.0 - float(np.std(scores)) if scores else 0.0
        confidence = np.clip(agreement, 0.0, 1.0)

        signal = AgentSignal(
            name=self.name,
            action=np.clip(action, -1.0, 1.0),
            confidence=confidence,
            metadata={
                "reddit": f"{reddit_score:.2f}",
                "news": f"{news_score:.2f}",
                "trends": f"{trends_score:.2f}",
                "fear_greed": fear_greed,
            },
        )
        self._last_signal = signal
        return signal


class OnChainAgent(BaseAgent):
    """Agent using on-chain data: whale movements, exchange flows, funding rates."""

    name = "onchain"

    def produce_signal(self, snapshot: MarketSnapshot, context: dict) -> AgentSignal:
        exchange_flow = context.get("exchange_flow", 0.0)  # negative = inflow (bearish)
        whale_alert = context.get("whale_alert", 0)
        funding_rate = context.get("funding_rate", 0.0)
        open_interest_change = context.get("oi_change", 0.0)

        # Exchange inflows (negative) = potential selling pressure
        flow_signal = np.clip(-exchange_flow, -1.0, 1.0)

        # Extreme positive funding = over-leveraged longs = reversal risk
        funding_signal = -np.clip(funding_rate * 10, -1.0, 1.0)

        # Whale alerts + OI trend agreement
        whale_oi_signal = np.clip(
            (whale_alert / 10.0 + open_interest_change) / 2.0,
            -1.0,
            1.0,
        )

        action = float(flow_signal * 0.4 + funding_signal * 0.3 + whale_oi_signal * 0.3)

        # Confidence from source agreement
        components = [flow_signal, funding_signal, whale_oi_signal]
        agreement = 1.0 - float(np.std(components))
        confidence = np.clip(agreement, 0.0, 1.0)

        signal = AgentSignal(
            name=self.name,
            action=np.clip(action, -1.0, 1.0),
            confidence=confidence,
            metadata={
                "exchange_flow": f"{exchange_flow:.3f}",
                "funding_rate": f"{funding_rate:.4f}",
                "whale_alert": whale_alert,
                "oi_change": f"{open_interest_change:.3f}",
            },
        )
        self._last_signal = signal
        return signal


class RiskAgent(BaseAgent):
    """Agent focused on risk assessment and market regime detection."""

    name = "risk"

    def produce_signal(self, snapshot: MarketSnapshot, context: dict) -> AgentSignal:
        # Market regime: trending vs ranging
        volatility = context.get("volatility_regime", "normal")
        correlation = context.get("btc_correlation", 0.0)
        volume_anomaly = context.get("volume_anomaly", False)

        risk_level = 0.0
        if volatility == "high":
            risk_level += 0.3
        if volatility == "extreme":
            risk_level += 0.5
        if abs(correlation) > 0.8:
            risk_level += 0.2
        if volume_anomaly:
            risk_level += 0.3

        # High risk = reduce position (toward flat)
        action = -np.clip(risk_level - 0.3, -1.0, 1.0)
        confidence = np.clip(risk_level, 0.0, 1.0)

        signal = AgentSignal(
            name=self.name,
            action=np.clip(action, -1.0, 1.0),
            confidence=confidence,
            metadata={
                "volatility_regime": volatility,
                "btc_correlation": f"{correlation:.2f}",
                "volume_anomaly": volume_anomaly,
                "risk_level": f"{risk_level:.2f}",
            },
        )
        self._last_signal = signal
        return signal


class AgentOrchestrator:
    """Coordinates multiple agents and produces a consensus signal."""

    def __init__(self, consensus_threshold: float = 0.1):
        self.agents: list[BaseAgent] = []
        self.consensus_threshold = consensus_threshold
        self._history: list[list[AgentSignal]] = []

    def register(self, agent: BaseAgent) -> None:
        self.agents.append(agent)

    def produce_consensus(
        self, snapshot: MarketSnapshot, context: dict
    ) -> Signal:
        if not self.agents:
            return Signal(
                action=0.0,
                confidence=0.0,
                snapshot=snapshot,
            )

        signals = []
        for agent in self.agents:
            try:
                sig = agent.produce_signal(snapshot, context)
                logger.info("Agent %s: %s", agent.name, sig)
                signals.append(sig)
            except Exception as e:
                logger.error("Agent %s failed: %s", agent.name, e)

        self._history.append(list(signals))

        # Weighted consensus
        if not signals:
            return Signal(action=0.0, confidence=0.0, snapshot=snapshot)

        total_weight = sum(a.weight for a in self.agents)
        if total_weight == 0:
            total_weight = len(signals)

        # Weighted action by agent weight * confidence
        weighted_action = 0.0
        weighted_confidence = 0.0
        weight_sum = 0.0

        for sig, agent in zip(signals, self.agents):
            w = agent.weight * sig.confidence
            weighted_action += sig.action * w
            weighted_confidence += sig.confidence * w
            weight_sum += w

        if weight_sum > 0:
            consensus_action = weighted_action / weight_sum
            consensus_confidence = weighted_confidence / weight_sum
        else:
            consensus_action = 0.0
            consensus_confidence = 0.0

        # Apply consensus threshold — below threshold, go flat
        if abs(consensus_action) < self.consensus_threshold:
            consensus_action = 0.0
            consensus_confidence = 0.0

        logger.info(
            "Consensus: action=%.3f, conf=%.2f, threshold=%.2f",
            consensus_action,
            consensus_confidence,
            self.consensus_threshold,
        )

        return Signal(
            action=np.clip(consensus_action, -1.0, 1.0),
            confidence=np.clip(consensus_confidence, 0.0, 1.0),
            snapshot=snapshot,
        )

    def update_weights_from_performance(self) -> None:
        """Adjust agent weights based on historical accuracy."""
        for agent in self.agents:
            # Weight = baseline 1.0 + accuracy bonus
            agent.weight = 0.5 + agent.accuracy
            logger.debug(
                "Updated %s weight to %.2f (accuracy=%.2f)",
                agent.name,
                agent.weight,
                agent.accuracy,
            )

    def get_agent_state(self) -> dict:
        """Current state of all agents."""
        return {
            agent.name: {
                "weight": agent.weight,
                "accuracy": agent.accuracy,
                "decisions": agent.total_decisions,
                "last_signal": str(agent._last_signal) if agent._last_signal else "none",
            }
            for agent in self.agents
        }
