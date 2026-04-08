"""FastAPI inference server for the execution engine.

Provides HTTP endpoints for:
- POST /v1/signal     → run model inference on provided features, return signal
- POST /v1/signal/live → use latest market data (fetches fresh OHLCV)
- GET  /v1/health     → engine status + model info
- GET  /v1/summary    → current portfolio state + risk metrics
- POST /v1/position/close → manually close open position
"""

from __future__ import annotations

import asyncio
import logging
import signal
import sys
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from typing import Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import uvicorn

from .config import Settings
from .data_feed import DataFeed
from .db import TradeDB
from .models import Side
from .paper_executor import PaperExecutor
from .risk_manager import RiskManager
from .signal_generator import Signal, SignalGenerator

logger = logging.getLogger(__name__)

# ── Global state (set by lifespan) ──────────────────────────────
engine: Optional[EngineFacade] = None


# ── Pydantic request/response models ────────────────────────────

class FeatureInput(BaseModel):
    """Pre-computed features for model inference."""
    features: dict[str, float] = Field(
        ...,
        description="Dictionary of feature name → value (must match training features)",
    )
    price: float = Field(..., gt=0, description="Current market price")


class SignalResponse(BaseModel):
    action: float
    side: str  # LONG / SHORT / FLAT
    confidence: float
    price: float
    timestamp: str
    executed: bool  # whether a trade was actually placed
    message: str = ""


class HealthResponse(BaseModel):
    status: str  # healthy / starting / error
    model_loaded: bool
    model_path: str
    symbol: str
    paper_trading: bool
    uptime_seconds: float
    position: Optional[dict]
    balance: float


class PositionCloseRequest(BaseModel):
    reason: str = "manual"


class PositionCloseResponse(BaseModel):
    success: bool
    pnl: float
    reason: str
    message: str


# ── Engine facade ───────────────────────────────────────────────

class EngineFacade:
    """Wire together all engine components for the API server."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.data_feed = DataFeed(settings)
        self.signal_gen = SignalGenerator(settings.model_path)
        self.risk = RiskManager(settings)
        self.db = TradeDB()
        self.executor = PaperExecutor(settings, self.risk, db=self.db)
        self._session_id = ""
        self._start_time = datetime.now(timezone.utc)

    def start(self) -> str:
        """Initialize and seed data."""
        import uuid
        self.executor.init(initial_balance=10_000.0)
        self._session_id = uuid.uuid4().hex[:16]
        self.executor.set_session_id(self._session_id)
        self.db.create_session(
            session_id=self._session_id,
            initial_balance=10_000.0,
            mode="paper" if self.settings.paper_trading else "live",
            model_path=self.settings.model_path,
        )
        return self._session_id

    def infer_from_features(self, features: dict[str, float], price: float) -> SignalResponse:
        """Run inference on pre-computed features."""
        snapshot = self.data_feed.compute_features_for_api(features, price)
        sig = self.signal_gen.predict(snapshot)
        executed = self._try_execute(sig)
        return self._to_response(sig, executed)

    async def infer_from_live(self) -> SignalResponse:
        """Fetch fresh OHLCV, compute features, run inference."""
        snapshot = await self.data_feed.refresh(None)
        sig = self.signal_gen.predict(snapshot)
        executed = self._try_execute(sig)
        return self._to_response(sig, executed)

    def close_position(self, reason: str = "manual") -> tuple[bool, float, str]:
        """Manually close an open position."""
        if self.executor.position is None:
            return False, 0.0, "No open position"
        price = self.executor.position.current_price
        pnl = self.executor._close_position(price, reason=reason)
        return True, pnl, f"Position closed, PnL: {pnl:+.2f}"

    def health(self) -> HealthResponse:
        pos = None
        if self.executor.position:
            p = self.executor.position
            pos = {
                "side": p.side.value,
                "quantity": round(p.quantity, 4),
                "entry_price": p.entry_price,
                "current_price": p.current_price,
                "unrealized_pnl": round(p.unrealized_pnl, 2),
            }
        uptime = (datetime.now(timezone.utc) - self._start_time).total_seconds()
        return HealthResponse(
            status="healthy",
            model_loaded=True,
            model_path=self.settings.model_path,
            symbol=self.settings.symbol,
            paper_trading=self.settings.paper_trading,
            uptime_seconds=round(uptime, 1),
            position=pos,
            balance=round(self.executor.balance, 2),
        )

    def summary(self) -> dict:
        return self.executor.summary()

    def _try_execute(self, sig: Signal) -> bool:
        order = self.executor.execute_signal(sig)
        return order is not None

    @staticmethod
    def _to_response(sig: Signal, executed: bool) -> SignalResponse:
        side = "LONG" if sig.is_long else ("SHORT" if sig.is_short else "FLAT")
        return SignalResponse(
            action=round(sig.action, 4),
            side=side,
            confidence=round(sig.confidence, 4),
            price=round(sig.price, 2),
            timestamp=sig.timestamp.isoformat(),
            executed=executed,
        )


# ── FastAPI app ─────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global engine
    settings = Settings()
    if not settings.model_path:
        logger.error("MODEL_PATH not set — server starting without model")
        yield
        return
    try:
        engine = EngineFacade(settings)
        sid = engine.start()
        logger.info("Engine started with session %s", sid)
    except Exception as e:
        logger.error("Failed to initialize engine: %s", e, exc_info=True)
    yield
    # Shutdown
    if engine:
        s = engine.executor.summary()
        engine.db.close_session(
            session_id=engine._session_id,
            final_equity=s["equity"],
            total_trades=s["total_trades"],
        )
        engine.db.close()
        logger.info("Engine shut down, session saved")


app = FastAPI(
    title="Autonomous Trading Engine",
    description="PPO RL model inference + paper trading execution",
    version="0.1.0",
    lifespan=lifespan,
)


@app.get("/health", response_model=HealthResponse)
@app.get("/v1/health", response_model=HealthResponse)
async def health():
    """Service health and basic status."""
    if engine is None:
        raise HTTPException(503, "Engine not initialized — check MODEL_PATH")
    return engine.health()


@app.get("/v1/summary")
async def summary():
    """Current portfolio state: balance, equity, risk metrics, recent trades."""
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    return engine.summary()


@app.post("/v1/signal", response_model=SignalResponse)
async def run_signal(body: FeatureInput):
    """Run model inference on pre-computed features.

    Supply the 12 features as a dict. The engine will run inference
    and optionally execute a trade if the signal is actionable.
    """
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    return engine.infer_from_features(body.features, body.price)


@app.post("/v1/signal/live", response_model=SignalResponse)
async def run_live_signal():
    """Fetch latest OHLCV from exchange, compute features, run inference."""
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    return await engine.infer_from_live()


@app.post("/v1/position/close", response_model=PositionCloseResponse)
async def close_position(body: PositionCloseRequest = PositionCloseRequest()):
    """Manually close any open position with the given reason."""
    if engine is None:
        raise HTTPException(503, "Engine not initialized")
    success, pnl, message = engine.close_position(reason=body.reason)
    return PositionCloseResponse(
        success=success,
        pnl=round(pnl, 2),
        reason=body.reason,
        message=message,
    )


def main() -> None:
    """CLI entry point for the API server."""
    logging.basicConfig(
        level=logging.DEBUG if "--verbose" in sys.argv else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    uvicorn.run(
        "src.server:app",
        host="0.0.0.0",
        port=8000,
        reload="--reload" in sys.argv,
    )


if __name__ == "__main__":
    main()
