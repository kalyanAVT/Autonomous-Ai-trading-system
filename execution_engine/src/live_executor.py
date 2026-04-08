"""Live trading executor — bridges paper trading to real exchange execution.

Provides a drop-in replacement for PaperExecutor with real order
placement via ccxt. Uses the same order types, risk checks, and
database logging.

CRITICAL: This executes real orders with real money.
Start with PAPER_TRADING=true in production.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Optional

import ccxt

from .config import Settings
from .db import TradeDB
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


class LiveExecutor:
    """Real exchange execution via ccxt.

    Mirrors PaperExecutor API for seamless swap between paper/live.
    """

    def __init__(
        self,
        settings: Settings,
        risk_manager: RiskManager,
        db: Optional[TradeDB] = None,
    ):
        self.settings = settings
        self.risk = risk_manager
        self.db = db
        self.balance = 0.0
        self.position: Position | None = None
        self.trade_history: list[TradeRecord] = []
        self.order_history: list[Order] = []
        self._session_id: str = ""

        # Initialize real exchange connection
        self.exchange = ccxt.binance({
            "apiKey": settings.api_key,
            "secret": settings.api_secret,
            "enableRateLimit": True,
            "options": {
                "defaultType": "spot",
                "sandbox": settings.use_testnet and not settings.paper_trading,
            },
        })
        logger.warning("LIVE EXECUTOR initialized with %s", settings.exchange_name)

    def init(self, initial_balance: float = 0.0) -> None:
        """Initialize by fetching real exchange balance."""
        if initial_balance > 0:
            self.balance = initial_balance
        else:
            self.balance = self._fetch_balance()
        self.risk.init_equity(self.balance)
        logger.info(
            "LiveExecutor initialized: balance=%.2f, symbol=%s",
            self.balance,
            self.settings.symbol,
        )

    def _fetch_balance(self) -> float:
        """Get real balance from exchange."""
        try:
            base_currency = self.settings.symbol.split("/")[1]  # e.g. USDT from BTC/USDT
            balance = self.exchange.fetch_balance()
            free = balance.get(base_currency, {}).get("free", 0)
            return float(free)
        except Exception as e:
            logger.error("Failed to fetch balance: %s", e)
            raise

    def execute_signal(self, signal: Signal) -> Order | None:
        """Process signal and execute real order.

        Same logic as PaperExecutor but sends real orders to exchange.
        """
        # Pre-trade risk check (same as paper)
        allowed, reason = self.risk.check_pre_trade(signal)
        if not allowed:
            logger.warning("Signal rejected by risk manager: %s", reason)
            return None

        if self.position is not None:
            # Check stop loss / take profit first
            self.position.update_price(signal.price)
            if self.check_stop_loss(signal.price, signal.price):
                return None
            if self.check_take_profit(signal.price, signal.price):
                return None

            # Only close if signal disagrees
            should_close = (
                (self.position.side == Side.LONG and not signal.is_long)
                or (self.position.side == Side.SHORT and not signal.is_short)
            )
            if should_close:
                self._close_position(signal.price)
                if signal.is_flat:
                    return None
            else:
                self.position.update_price(signal.price)
                return None

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
        """Open real position on exchange."""
        # Calculate position size (same logic as paper)
        exposure = abs(signal.action) * self.balance * self.settings.max_position_pct
        commission = exposure * self.settings.commission_pct

        total_cost = exposure + commission
        if total_cost > self.balance:
            logger.warning(
                "Insufficient balance: exposure=%.2f + commission=%.2f > balance=%.2f",
                exposure, commission, self.balance,
            )
            raise ValueError(
                f"Insufficient balance: need {total_cost:.2f}, have {self.balance:.2f}"
            )

        # Place real market order
        quantity = exposure / signal.price if signal.price > 0 else 0.0
        direction = "buy" if signal.action > 0 else "sell"

        logger.info(
            "Placing %s MARKET order: %s @ %s",
            direction, quantity, signal.price,
        )

        try:
            result = self.exchange.create_market_order(
                symbol=self.settings.symbol,
                side=direction,
                amount=quantity,
            )
        except ccxt.InsufficientFunds as e:
            logger.error("Insufficient funds on exchange: %s", e)
            raise ValueError(str(e))
        except ccxt.ExchangeError as e:
            logger.error("Exchange error: %s", e)
            raise ValueError(f"Exchange error: {e}")

        # Parse fill
        fill_price = float(result.get("average", result.get("price", signal.price)))
        fill_qty = float(result.get("filled", quantity))
        exchange_fee = float(result.get("fee", {}).get("cost", commission))

        order = Order(
            symbol=self.settings.symbol,
            side=direction,
            order_type=OrderType.MARKET,
            quantity=fill_qty,
            fill_price=fill_price,
            commission=exchange_fee,
            status=OrderStatus.FILLED,
            filled_at=datetime.now(timezone.utc),
        )

        # Stop-loss and take-profit
        stop_loss_price = (
            fill_price * (1.0 - self.settings.stop_loss_pct)
            if signal.action > 0
            else fill_price * (1.0 + self.settings.stop_loss_pct)
        )
        take_profit_price = (
            fill_price * (1.0 + self.settings.take_profit_pct)
            if signal.action > 0
            else fill_price * (1.0 - self.settings.take_profit_pct)
        )

        position = Position(
            symbol=self.settings.symbol,
            side=Side.LONG if signal.action > 0 else Side.SHORT,
            quantity=fill_qty,
            entry_price=fill_price,
            current_price=fill_price,
            stop_loss=stop_loss_price,
            take_profit=take_profit_price,
        )
        self.position = position
        self.balance -= exchange_fee
        self.order_history.append(order)

        logger.info(
            "Opened %s: qty=%.4f entry=%.2f commission=%.2f order_id=%s",
            order.side.value, fill_qty, fill_price, exchange_fee, result.get("id"),
        )
        return order

    def _close_position(self, current_price: float, reason: str = "signal") -> float:
        """Close position with real exchange order."""
        if self.position is None:
            return 0.0

        pos = self.position
        close_side = "sell" if pos.side == Side.LONG else "buy"

        logger.info(
            "Closing position: %s %.4f %s @ %s",
            close_side, pos.quantity, pos.symbol, current_price,
        )

        try:
            result = self.exchange.create_market_order(
                symbol=self.settings.symbol,
                side=close_side,
                amount=pos.quantity,
            )
        except ccxt.ExchangeError as e:
            logger.error("Error closing position: %s", e)
            return 0.0

        fill_price = float(result.get("average", result.get("price", current_price)))
        exchange_fee = float(result.get("fee", {}).get("cost", 0))

        # Calculate PnL
        if pos.side == Side.LONG:
            pnl = (fill_price - pos.entry_price) * pos.quantity - exchange_fee
        else:
            pnl = (pos.entry_price - fill_price) * pos.quantity - exchange_fee

        self.balance -= exchange_fee

        holding_hours = (
            (datetime.now(timezone.utc) - pos.entry_time).total_seconds() / 3600.0
        )

        trade = TradeRecord(
            symbol=pos.symbol,
            side=pos.side,
            entry_price=pos.entry_price,
            exit_price=fill_price,
            quantity=pos.quantity,
            pnl=pnl,
            commission=exchange_fee,
            slippage=0.0,
            entry_time=pos.entry_time,
            exit_time=datetime.now(timezone.utc),
            holding_periods_hours=holding_hours,
            reason=reason,
        )

        self.trade_history.append(trade)
        stop_loss = pos.stop_loss
        exposure_pct = 0.0
        self.risk.on_trade_closed(pnl, exposure_pct)
        self.risk.update_equity(self.balance)
        self._log_trade_to_db(trade, stop_loss)

        self.position = None
        logger.info("Closed %s: pnl=%.2f balance=%.2f", pos.side.value, pnl, self.balance)
        return pnl

    def check_stop_loss(self, low: float, high: float) -> bool:
        if self.position is None or self.position.stop_loss is None:
            return False
        if self.position.side == Side.LONG and low <= self.position.stop_loss:
            logger.warning(
                "STOP LOSS TRIGGERED: price=%.2f stop=%.2f", low, self.position.stop_loss
            )
            self._close_position(self.position.stop_loss, reason="stop_loss")
            return True
        elif self.position.side == Side.SHORT and high >= self.position.stop_loss:
            logger.warning(
                "STOP LOSS TRIGGERED: price=%.2f stop=%.2f", high, self.position.stop_loss
            )
            self._close_position(self.position.stop_loss, reason="stop_loss")
            return True
        return False

    def check_take_profit(self, low: float, high: float) -> bool:
        if self.position is None or self.position.take_profit is None:
            return False
        if self.position.side == Side.LONG and high >= self.position.take_profit:
            logger.info(
                "TAKE PROFIT: price=%.2f tp=%.2f", high, self.position.take_profit
            )
            self._close_position(self.position.take_profit, reason="take_profit")
            return True
        elif self.position.side == Side.SHORT and low <= self.position.take_profit:
            self._close_position(self.position.take_profit, reason="take_profit")
            return True
        return False

    @property
    def equity(self) -> float:
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

    def set_session_id(self, session_id: str) -> None:
        self._session_id = session_id

    def _log_trade_to_db(self, trade: TradeRecord, stop_loss: Optional[float]) -> None:
        if not self.db or not self._session_id:
            return
        try:
            self.db.log_trade(
                session_id=self._session_id,
                symbol=trade.symbol,
                side=trade.side.value,
                entry_price=trade.entry_price,
                exit_price=trade.exit_price,
                quantity=trade.quantity,
                pnl=trade.pnl,
                commission=trade.commission,
                stop_loss=stop_loss,
                take_profit=None,
                reason=trade.reason,
                entry_time=trade.entry_time,
                exit_time=trade.exit_time,
                holding_hours=trade.holding_periods_hours,
            )
        except Exception as e:
            logger.error("Failed to log trade to DB: %s", e)
