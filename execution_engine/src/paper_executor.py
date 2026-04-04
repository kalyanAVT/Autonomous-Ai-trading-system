"""Paper trading execution engine.

Simulates order execution against real-time market prices with
realistic slippage and commission modeling. No real money at risk.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import numpy as np

from .config import Settings
from .models import (
    Order,
    OrderStatus,
    OrderType,
    Position,
    Side,
    TradeRecord,
)
from .risk_manager import RiskManager
from .signal_generator import Signal

logger = logging.getLogger(__name__)


class PaperExecutor:
    """Simulated exchange for paper trading."""

    def __init__(self, settings: Settings, risk_manager: RiskManager):
        self.settings = settings
        self.risk = risk_manager
        self.balance = 0.0
        self.position: Position | None = None
        self.trade_history: list[TradeRecord] = []
        self.order_history: list[Order] = []

    def init(self, initial_balance: float = 10_000.0) -> None:
        """Initialize paper account."""
        self.balance = initial_balance
        self.risk.init_equity(initial_balance)
        logger.info(
            "PaperExecutor initialized: balance=%.2f, symbol=%s",
            initial_balance,
            self.settings.symbol,
        )

    def execute_signal(self, signal: Signal) -> Order | None:
        """Process a model signal and execute the corresponding order.

        Returns the Order if executed, None if no action taken.
        """
        # Pre-trade risk check
        allowed, reason = self.risk.check_pre_trade(signal)
        if not allowed:
            logger.warning("Signal rejected by risk manager: %s", reason)
            return None

        if self.position is not None:
            self._close_position(signal.price)

        if signal.is_flat:
            return None

        side = Side.LONG if signal.is_long else Side.SHORT
        try:
            order = self._open_position(side, signal)
        except ValueError as e:
            logger.error("Failed to open position: %s", e)
            return None
        return order

    def _open_position(self, side: Side, signal: Signal) -> Order:
        """Open a new position based on signal."""
        exposure = abs(signal.action) * self.balance * self.settings.max_position_pct
        price = self._apply_slippage(signal.price, side)
        commission = exposure * self.settings.commission_pct

        # Guard: ensure balance never goes negative
        total_cost = exposure + commission
        if total_cost > self.balance:
            logger.warning(
                "Insufficient balance: exposure=%.2f + commission=%.2f > balance=%.2f, "
                "skipping order",
                exposure, commission, self.balance,
            )
            raise ValueError(
                f"Insufficient balance for position: need {total_cost:.2f}, have {self.balance:.2f}"
            )

        slippage_cost = abs(price - signal.price) * (exposure / signal.price)
        quantity = exposure / price if price > 0 else 0.0
        self.balance -= total_cost

        direction = Side.LONG if signal.action > 0 else Side.SHORT

        order = Order(
            symbol=self.settings.symbol,
            side=direction,
            order_type=OrderType.MARKET,
            quantity=quantity,
            fill_price=price,
            commission=commission,
            slippage=slippage_cost,
            status=OrderStatus.FILLED,
            filled_at=datetime.now(timezone.utc),
        )

        position = Position(
            symbol=self.settings.symbol,
            side=direction,
            quantity=quantity,
            entry_price=price,
            current_price=price,
            stop_loss=None,
        )
        self.position = position
        self.order_history.append(order)

        logger.info(
            "Opened %s: qty=%.4f entry=%.2f commission=%.2f",
            direction.value,
            quantity,
            price,
            commission,
        )
        return order

    def _close_position(self, current_price: float) -> float:
        """Close current position and return realized PnL."""
        if self.position is None:
            return 0.0

        pos = self.position
        commission = (pos.quantity * current_price) * self.settings.commission_pct

        if pos.side == Side.LONG:
            pnl = (current_price - pos.entry_price) * pos.quantity - commission
        else:
            pnl = (pos.entry_price - current_price) * pos.quantity - commission

        # Return capital to balance
        self.balance += pos.quantity * current_price - commission

        holding_hours = (
            (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600.0
        )

        trade = TradeRecord(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=current_price,
            quantity=pos.quantity,
            pnl=pnl,
            commission=commission,
            slippage=0.0,
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc),
            holding_periods_hours=holding_hours,
            reason="signal",
        )

        self.trade_history.append(trade)
        exposure_pct = 0.0
        self.risk.on_trade_closed(pnl, exposure_pct)
        self.risk.update_equity(self.balance)

        self.position = None

        logger.info(
            "Closed %s: pnl=%.2f balance=%.2f",
            trade.side.value,
            pnl,
            self.balance,
        )
        return pnl

    def check_stop_loss(self, low: float, high: float) -> bool:
        """Check if stop loss was triggered by current candle."""
        if self.position is None or self.position.stop_loss is None:
            return False

        if self.position.side == Side.LONG and low <= self.position.stop_loss:
            logger.info(
                "Stop loss hit: price=%.2f stop=%.2f", low, self.position.stop_loss
            )
            self._close_position(self.position.stop_loss)
            return True
        elif self.position.side == Side.SHORT and high >= self.position.stop_loss:
            self._close_position(self.position.stop_loss)
            return True

        return False

    @property
    def equity(self) -> float:
        """Current total equity = balance + unrealized position value."""
        if self.position is None:
            return self.balance
        if self.position.side == Side.LONG:
            return self.balance + self.position.quantity * self.position.current_price
        else:
            return (
                self.balance
                + self.position.quantity * self.position.entry_price
                - self.position.quantity * self.position.current_price
            )

    def summary(self) -> dict:
        """Return portfolio summary."""
        return {
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "position": (
                {
                    "side": self.position.side.value,
                    "quantity": round(self.position.quantity, 4),
                    "entry_price": self.position.entry_price,
                    "current_price": self.position.current_price,
                    "unrealized_pnl": round(self.position.unrealized_pnl, 2),
                }
                if self.position
                else "flat"
            ),
            "total_trades": len(self.trade_history),
            "net_pnl": round(sum(t.pnl for t in self.trade_history), 2),
            "risk_state": self.risk.daily_summary(),
        }

    def _apply_slippage(self, price: float, side: Side) -> float:
        """Apply realistic slippage based on trade direction."""
        slip = float(np.random.uniform(0, self.settings.slippage_pct))
        if side == Side.LONG:
            return price * (1.0 + slip)  # Buy at worse price
        else:
            return price * (1.0 - slip)  # Sell at worse price
