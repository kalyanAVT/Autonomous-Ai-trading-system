"""Main entry point for the execution engine.

Wires together: data feed → signal fusion → risk check → paper/live execution.
Signal fusion combines:
  - PPO model signals (technical)
  - Multi-agent consensus (sentiment, on-chain, risk)
  - Social intelligence (Reddit, Fear & Greed, market data)
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .data_feed import DataFeed
from .db import TradeDB
from .paper_executor import PaperExecutor
from .risk_manager import RiskManager
from .signal_generator import SignalGenerator
from .signal_fusion import SignalFusion
from .agents import (
    AgentOrchestrator,
    TechnicalAgent,
    SentimentAgent,
    OnChainAgent,
    RiskAgent,
)
from .social_intel import (
    SocialIntelligence,
    OnChainIntelligence,
    MarketRegimeDetector,
)

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Top-level coordination of all engine components."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_feed = DataFeed(settings)
        self.signal_gen = SignalGenerator(settings.model_path)
        self.risk = RiskManager(settings)
        self.db = TradeDB()
        self.executor = PaperExecutor(settings, self.risk, db=self.db)

        # Signal fusion
        self.fusion = SignalFusion(
            use_agent_consensus=settings.use_agent_consensus,
            use_direct_model=settings.use_direct_model,
            agent_weight=settings.agent_fusion_weight,
            model_weight=settings.model_fusion_weight,
            min_confidence_threshold=settings.min_confidence_threshold,
        )

        # Agent orchestration
        self.agents = AgentOrchestrator(
            consensus_threshold=settings.consensus_threshold,
        )
        self.agents.register(
            TechnicalAgent(weight=1.2)  # Slightly higher weight for model-based
        )
        self.agents.register(SentimentAgent(weight=1.0))
        self.agents.register(OnChainAgent(weight=1.0))
        self.agents.register(RiskAgent(weight=0.8))  # Risk is veto-like, lower weight

        # Social intelligence
        self.social_intel = SocialIntelligence(
            cryptopanic_api=settings.cryptopanic_api,
        )
        self.onchain_intel = OnChainIntelligence(
            symbol=settings.symbol_onchain,
            coingecko_id=settings.coingecko_id,
        )
        self._last_social_context: dict = {}

        self._session_id = ""
        self._running = False
        self._social_last_collected = 0.0
        self._last_prices: list[float] = []

    async def run(self) -> None:
        """Start the execution loop."""
        self._running = True
        self.executor.init(initial_balance=10_000.0)
        session_id = uuid.uuid4().hex[:16]
        self._session_id = session_id
        self.executor.set_session_id(session_id)
        self.db.create_session(
            session_id=session_id,
            initial_balance=10_000.0,
            mode="paper" if self.settings.paper_trading else "live",
            model_path=self.settings.model_path,
        )
        logger.info("Session %s started", session_id)

        logger.info("=== Starting Execution Engine ===")
        logger.info("Symbol: %s | Timeframe: %s | Paper: %s",
                     self.settings.symbol, self.settings.timeframe, self.settings.paper_trading)
        logger.info(
            "Mode: agents=%s, model=%s, fusion=%.0f/%.0f",
            self.settings.use_agent_consensus,
            self.settings.use_direct_model,
            self.settings.agent_fusion_weight,
            self.settings.model_fusion_weight,
        )

        # Seed historical data
        await self.data_feed.fetch_history()
        logger.info("Historical data seeded, entering signal loop")

        # Initial social context collection
        self._last_social_context = await self._collect_context()

        loop_interval = self.settings.poll_interval_seconds

        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("Error in execution loop: %s", e, exc_info=True)

            await self._sleep(loop_interval)

        logger.info("=== Execution Engine Stopped ===")
        self._print_final_summary()

    async def _tick(self) -> None:
        """Single loop iteration: fetch → signal fusion → execute."""
        logger.info("--- Tick: %s UTC ---", datetime.now(timezone.utc).isoformat())

        # If we have an open position, check stop loss first
        if self.executor.position is not None:
            try:
                latest = await self.data_feed.refresh(None, n_candles=1)
                self.executor.position.update_price(latest.close)
                current_price = self.executor.position.current_price
                self.executor.check_stop_loss(
                    low=current_price, high=current_price
                )
                self.executor.check_take_profit(
                    low=current_price, high=current_price
                )
            except Exception:
                pass

        # Refresh market data and compute features
        snapshot = await self.data_feed.refresh(None)

        # Track price history for regime detection
        self._last_prices.append(snapshot.close)
        if len(self._last_prices) > 100:
            self._last_prices = self._last_prices[-100:]

        # Collect social/on-chain context periodically
        now = datetime.now(timezone.utc).timestamp()
        if now - self._social_last_collected >= self.settings.social_poll_interval:
            self._last_social_context = await self._collect_context()
            self._social_last_collected = now

        # Build context for agents
        context = dict(self._last_social_context)
        if self._last_prices:
            volumes = [1.0] * len(self._last_prices)  # Use normalized volume
            regime = MarketRegimeDetector.detect(
                prices=self._last_prices,
                volumes=volumes,
            )
            context.update(regime)

        # Update technical agent with fresh model signal
        model_signal = self.signal_gen.predict(snapshot)
        self.agents.agents[0].model_signal = model_signal  # TechnicalAgent is first

        # Generate agent consensus
        agent_consensus = self.agents.produce_consensus(snapshot, context)

        # Fuse all signals
        fused_signal = self.fusion.fuse(
            snapshot=snapshot,
            agent_signal=agent_consensus,
            model_signal=model_signal,
            context=context,
        )
        logger.info(str(fused_signal))
        logger.info("Agent states: %s", self.agents.get_agent_state())

        # Execute against paper/live exchange
        order = self.executor.execute_signal(fused_signal)
        if order:
            logger.info("Order executed: %s", order)

        # Log current state
        summary = self.executor.summary()
        logger.info(
            "Equity: %.2f | Position: %s | Trades: %d | Net PnL: %.2f",
            summary["equity"],
            str(summary["position"]),
            summary["total_trades"],
            summary["net_pnl"],
        )

    async def _collect_context(self) -> dict:
        """Collect social and on-chain intelligence concurrently."""
        social_task = asyncio.create_task(self.social_intel.collect_all())
        onchain_task = asyncio.create_task(self.onchain_intel.collect_all())
        social_result, onchain_result = await asyncio.gather(
            social_task, onchain_task, return_exceptions=True
        )

        context = {}
        if isinstance(social_result, dict):
            context.update(social_result)
        if isinstance(onchain_result, dict):
            context.update(onchain_result)

        logger.info("Context collected: %s", context)
        return context

    def _print_final_summary(self) -> None:
        """Print session summary on shutdown."""
        s = self.executor.summary()
        r = s["risk_state"]
        print(f"\n{'='*50}")
        print(f"        SESSION SUMMARY")
        print(f"{'='*50}")
        print(f"Initial balance: $10,000.00")
        print(f"Final equity:    ${s['equity']:,.2f}")
        print(f"Net PnL:         ${s['net_pnl']:,.2f}")
        print(f"Total trades:    {s['total_trades']}")
        print(f"Win rate:        {r['win_rate']*100:.1f}%")
        print(f"Largest win:     ${r['largest_win']:,.2f}")
        print(f"Largest loss:    ${r['largest_loss']:,.2f}")
        print(f"Max drawdown:    {r['max_drawdown_pct']:.2f}%")
        print(f"Circuit breaker: {r['circuit_breaker']}")
        print(f"{'='*50}\n")
        # Persist session closure
        self.db.close_session(
            session_id=self._session_id,
            final_equity=s["equity"],
            total_trades=s["total_trades"],
        )
        self.db.close()

    def stop(self) -> None:
        self._running = False

    @staticmethod
    async def _sleep(seconds: int) -> None:
        """Interruptible sleep."""
        try:
            await asyncio.sleep(seconds)
        except asyncio.CancelledError:
            pass


def main() -> None:
    """CLI entry point."""
    logging.basicConfig(
        level=logging.DEBUG if "--verbose" in sys.argv else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = Settings()
    if not settings.model_path:
        logger.error("MODEL_PATH not set. Pass via --model flag or .env file")
        sys.exit(1)

    engine = ExecutionEngine(settings)

    loop = asyncio.new_event_loop()

    def _shutdown(sig: int, frame: object) -> None:
        logger.info("Received %s, shutting down...", sig)
        engine.stop()

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        loop.run_until_complete(engine.run())
    except KeyboardInterrupt:
        engine.stop()
    finally:
        loop.close()


if __name__ == "__main__":
    main()
