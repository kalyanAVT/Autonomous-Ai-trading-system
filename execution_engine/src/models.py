"""Core domain types for the execution engine."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Optional
from uuid import uuid4


class Side(str, Enum):
    LONG = "long"
    SHORT = "short"


class OrderType(str, Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP_LOSS = "stop_loss"


class OrderStatus(str, Enum):
    PENDING = "pending"
    FILLED = "filled"
    CANCELLED = "cancelled"
    REJECTED = "rejected"


@dataclass(frozen=True)
class OHLCVCandle:
    """Single OHLCV candle."""

    timestamp: datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass(frozen=True)
class MarketSnapshot:
    """Point-in-time market data with derived features."""

    timestamp: datetime
    symbol: str
    close: float
    features: Optional[dict[str, float]] = None


@dataclass
class Order:
    """Represents a trading order."""

    symbol: str
    side: Side
    order_type: OrderType
    quantity: float
    price: Optional[float] = None  # For limit/stop orders
    stop_price: Optional[float] = None  # For stop orders
    status: OrderStatus = OrderStatus.PENDING
    order_id: str = field(default_factory=lambda: uuid4().hex)
    created_at: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc), hash=False
    )
    filled_at: Optional[datetime] = None
    fill_price: Optional[float] = None
    commission: float = 0.0
    slippage: float = 0.0

    def fill(self, fill_price: float, commission: float = 0.0, slippage: float = 0.0) -> None:
        """Mark order as filled."""
        object.__setattr__(self, "status", OrderStatus.FILLED)
        object.__setattr__(self, "fill_price", fill_price)
        object.__setattr__(self, "commission", commission)
        object.__setattr__(self, "slippage", slippage)
        object.__setattr__(self, "filled_at", datetime.now(timezone.utc))


@dataclass
class Position:
    """Current portfolio position."""

    symbol: str
    side: Side
    quantity: float
    entry_price: float
    current_price: float
    unrealized_pnl: float = 0.0
    realized_pnl: float = 0.0
    entry_time: datetime = field(
        default_factory=lambda: datetime.now(timezone.utc), hash=False
    )
    stop_loss: Optional[float] = None

    def update_price(self, price: float) -> None:
        object.__setattr__(self, "current_price", price)
        if self.side == Side.LONG:
            object.__setattr__(
                self, "unrealized_pnl", (price - self.entry_price) * self.quantity
            )
        else:
            object.__setattr__(
                self, "unrealized_pnl", (self.entry_price - price) * self.quantity
            )


@dataclass(frozen=True)
class TradeRecord:
    """Immutable record of a completed trade."""

    symbol: str
    side: Side
    entry_price: float
    exit_price: float
    quantity: float
    pnl: float
    commission: float
    slippage: float
    entry_time: datetime
    exit_time: datetime
    holding_periods_hours: float
    reason: str  # "signal", "stop_loss", "circuit_breaker", "end_of_data"


@dataclass
class RiskState:
    """Current risk state of the portfolio."""

    total_equity: float
    daily_pnl: float = 0.0
    daily_return_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    current_drawdown_pct: float = 0.0
    position_exposure_pct: float = 0.0
    peak_equity: float = 0.0
    consecutive_losses: int = 0
