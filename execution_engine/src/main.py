"""Main entry point for the execution engine.

Wires together: data feed → model inference → risk check → paper execution.
Runs a continuous loop polling for new signals and executing trades.
"""

from __future__ import annotations

import asyncio
import json
import logging
import signal
import sys
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings
from .data_feed import DataFeed
from .paper_executor import PaperExecutor
from .risk_manager import RiskManager
from .signal_generator import SignalGenerator

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """Top-level coordination of all engine components."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.data_feed = DataFeed(settings)
        self.signal_gen = SignalGenerator(settings.model_path)
        self.risk = RiskManager(settings)
        self.executor = PaperExecutor(settings, self.risk)
        self._running = False

    async def run(self) -> None:
        """Start the execution loop."""
        self._running = True
        self.executor.init(initial_balance=10_000.0)

        logger.info("=== Starting Execution Engine ===")
        logger.info("Symbol: %s | Timeframe: %s | Paper: %s",
                     self.settings.symbol, self.settings.timeframe, self.settings.paper_trading)

        # Seed historical data
        await self.data_feed.fetch_history()
        logger.info("Historical data seeded, entering signal loop")

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
        """Single loop iteration: fetch → signal → execute."""
        logger.info("--- Tick: %s UTC ---", datetime.now(timezone.utc).isoformat())

        # If we have an open position, check stop loss first
        if self.executor.position is not None and self.executor.position.stop_loss:
            latest = await self.data_feed.refresh(
                None, n_candles=1
            )
            self.executor.position.update_price(latest.close)

        # Refresh market data and compute features
        snapshot = await self.data_feed.refresh(None)

        # Generate signal from model
        sig = self.signal_gen.predict(snapshot)
        logger.info(str(sig))

        # Execute against paper exchange
        order = self.executor.execute_signal(sig)
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
