"""Risk management layer for the execution engine.

Enforces constraints before any order reaches the broker/exchange:
- Max position size per trade
- Daily loss limits
- Maximum portfolio drawdown
- Consecutive loss circuit breaker
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from .config import Settings
from .models import RiskState
from .signal_generator import Signal

logger = logging.getLogger(__name__)


@dataclass
class RiskStats:
    """Accumulated risk statistics for reporting."""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_pnl: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    last_trade_time: datetime | None = None
    daily_start_equity: float = 0.0

    def record_trade(self, pnl: float, equity: float) -> None:
        self.total_trades += 1
        self.total_pnl += pnl
        if pnl > 0:
            self.winning_trades += 1
            self.largest_win = max(self.largest_win, pnl)
        else:
            self.losing_trades += 1
            self.largest_loss = min(self.largest_loss, pnl)
        self.last_trade_time = datetime.now(timezone.utc)
        self.daily_start_equity = equity


class RiskManager:
    """Pre-trade risk checks and portfolio protection."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self._state = RiskState(
            total_equity=0.0,
            peak_equity=0.0,
        )
        self._stats = RiskStats()
        self._circuit_breaker_active = False

    def init_equity(self, initial: float) -> None:
        """Set initial equity and record baseline for daily tracking."""
        self._state.total_equity = initial
        self._state.peak_equity = initial
        self._stats.daily_start_equity = initial
        logger.info("RiskManager initialized with equity=%.2f", initial)

    @property
    def state(self) -> RiskState:
        return self._state

    def update_equity(self, new_equity: float) -> None:
        """Update current equity and recalculate drawdown."""
        prev = self._state.total_equity
        self._state.total_equity = new_equity

        if new_equity > self._state.peak_equity:
            self._state.peak_equity = new_equity

        if self._state.peak_equity > 0:
            dd = (self._state.peak_equity - new_equity) / self._state.peak_equity
            self._state.current_drawdown_pct = dd
            self._state.max_drawdown_pct = max(self._state.max_drawdown_pct, dd)

        if self._stats.daily_start_equity > 0:
            self._state.daily_return_pct = (
                (new_equity - self._stats.daily_start_equity)
                / self._stats.daily_start_equity
            )
            self._state.daily_pnl = new_equity - self._stats.daily_start_equity

        logger.debug(
            "Equity: %.2f | DD: %.2f%% | Daily PnL: %.2f",
            new_equity,
            self._state.current_drawdown_pct * 100,
            self._state.daily_pnl,
        )

    def check_pre_trade(self, signal: Signal) -> tuple[bool, str]:
        """Validate a signal against risk limits before execution.

        Returns (allowed, reason).
        """
        # Circuit breaker
        if self._circuit_breaker_active:
            return False, "Circuit breaker active — trading halted"

        # Daily loss limit
        if self._state.daily_return_pct < -self.settings.daily_loss_limit_pct:
            self._activate_circuit_breaker(
                f"Daily loss limit hit: {self._state.daily_return_pct*100:.2f}%"
            )
            return False, "Daily loss limit exceeded"

        # Max drawdown
        if self._state.max_drawdown_pct >= self.settings.max_drawdown_pct:
            self._activate_circuit_breaker(
                f"Max drawdown hit: {self._state.max_drawdown_pct*100:.2f}%"
            )
            return False, "Maximum drawdown exceeded"

        # Consecutive losses
        if self._state.consecutive_losses >= self.settings.max_consecutive_losses:
            self._activate_circuit_breaker(
                f"Consecutive losses: {self._state.consecutive_losses}"
            )
            return False, "Consecutive loss limit exceeded"

        return True, "OK"

    def on_trade_closed(
        self, pnl: float, position_exposure_pct: float
    ) -> None:
        """Update risk state after a trade is closed."""
        self._stats.record_trade(pnl, self._state.total_equity)
        self._state.position_exposure_pct = position_exposure_pct

        if pnl <= 0:
            self._state.consecutive_losses += 1
        else:
            self._state.consecutive_losses = 0

    def _activate_circuit_breaker(self, reason: str) -> None:
        self._circuit_breaker_active = True
        logger.critical("CIRCUIT BREAKER: %s", reason)

    def reset_circuit_breaker(self) -> None:
        self._circuit_breaker_active = False
        self._state.consecutive_losses = 0
        logger.info("Circuit breaker reset")

    @property
    def is_halted(self) -> bool:
        return self._circuit_breaker_active

    def daily_summary(self) -> dict:
        return {
            "total_trades": self._stats.total_trades,
            "winning_trades": self._stats.winning_trades,
            "losing_trades": self._stats.losing_trades,
            "win_rate": (
                self._stats.winning_trades / self._stats.total_trades
                if self._stats.total_trades > 0
                else 0.0
            ),
            "total_pnl": round(self._stats.total_pnl, 2),
            "largest_win": round(self._stats.largest_win, 2),
            "largest_loss": round(self._stats.largest_loss, 2),
            "current_equity": round(self._state.total_equity, 2),
            "daily_return_pct": round(self._state.daily_return_pct * 100, 4),
            "current_drawdown_pct": round(self._state.current_drawdown_pct * 100, 4),
            "max_drawdown_pct": round(self._state.max_drawdown_pct * 100, 4),
            "circuit_breaker": self._circuit_breaker_active,
        }
