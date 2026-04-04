"""Backtesting engine.

Replays pre-computed historical features through the full execution pipeline:
SignalGenerator (PPO) → RiskManager → PaperExecutor.

Produces detailed performance metrics and trade logs without needing
live market connectivity or re-computing features.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np

from .config import Settings
from .models import MarketSnapshot
from .paper_executor import PaperExecutor
from .risk_manager import RiskManager
from .signal_generator import FEATURE_COLUMNS, Signal, SignalGenerator

logger = logging.getLogger(__name__)


# Index column name in the parquet data, used to retrieve bar timestamps.
_TIMESTAMP_COLUMN = "timestamp"


@dataclass
class BacktestResult:
    """Complete backtest outcome."""

    strategy: str
    symbol: str
    total_bars: int
    bars_traded: int
    initial_balance: float
    final_equity: float
    total_return_pct: float
    annualized_return_pct: float
    max_drawdown_pct: float
    sharpe_ratio: float
    sortino_ratio: float
    win_rate: float
    profit_factor: float
    total_trades: int
    avg_trade_holding_hours: float
    trades: list[dict] = field(default_factory=list)


class Backtester:
    """Replays historical features through the execution pipeline."""

    def __init__(
        self,
        signal_gen: SignalGenerator,
        settings: Settings,
        initial_balance: float = 10_000.0,
    ):
        self.signal_gen = signal_gen
        self.settings = settings
        self.initial_balance = initial_balance
        self.risk = RiskManager(settings)
        self.executor = PaperExecutor(settings, self.risk)
        self._bars_traded: int = 0

    def run(self, data_path: str, prices_path: str) -> BacktestResult:
        """Run backtest on pre-computed parquet features.

        Args:
            data_path: Path to features parquet (already computed by research_lab).
            prices_path: Path to aligned close prices .npy.

        Returns:
            BacktestResult with full performance breakdown.
        """
        import pandas as pd

        data = pd.read_parquet(data_path)
        prices = np.load(prices_path)

        assert len(data) == len(prices), (
            f"Feature rows ({len(data)}) != prices ({len(prices)})"
        )

        logger.info(
            "Backtesting on %d bars, %d features, symbol=%s",
            len(data),
            len(data.columns),
            self.settings.symbol,
        )

        self.executor.init(initial_balance=self.initial_balance)

        # Convert features to numpy array for fast inference
        feature_matrix = data[FEATURE_COLUMNS].values.astype(np.float32)

        # Resolve timestamps: prefer the parquet index, fall back to a column.
        if _TIMESTAMP_COLUMN in data.columns:
            timestamps = pd.to_datetime(data[_TIMESTAMP_COLUMN], utc=True)
        elif isinstance(data.index, pd.DatetimeIndex):
            timestamps = data.index if data.index.tzinfo else data.index.tz_localize("UTC")
        else:
            timestamps = pd.DatetimeIndex(
                [datetime.now(timezone.utc)] * len(data)
            )

        for i in range(len(feature_matrix)):
            row = feature_matrix[i]
            close = float(prices[i])

            # Clip like training
            clipped = np.clip(row.reshape(1, -1), -10.0, 10.0)
            action, _ = self.signal_gen.model.predict(clipped, deterministic=True)

            ts = timestamps[i].to_pydatetime()
            feature_dict = {FEATURE_COLUMNS[j]: float(row[j]) for j in range(len(row))}

            signal = Signal(
                action=float(action[0]),
                confidence=float(abs(action[0])),
                snapshot=MarketSnapshot(
                    timestamp=ts,
                    symbol=self.settings.symbol,
                    close=close,
                    features=feature_dict,
                ),
            )

            # Update position mark-to-market
            if self.executor.position is not None:
                self.executor.position.update_price(close)

            # Execute signal through paper executor
            self.executor.execute_signal(signal)
            self._bars_traded += 1

            # Update equity and risk
            eq = self.executor.equity
            self.risk.update_equity(eq)

            if self.risk.is_halted:
                logger.info("Trading halted at bar %d/%d", i, len(feature_matrix))
                break

        return self._compute_results(len(data))

    def _compute_results(self, total_bars: int) -> BacktestResult:
        """Compute final performance metrics."""
        trades = self.executor.trade_history
        wins = [t for t in trades if t.pnl > 0]
        losses = [t for t in trades if t.pnl <= 0]

        gross_profit = sum(t.pnl for t in wins)
        gross_loss = abs(sum(t.pnl for t in losses)) if losses else 1.0

        # Per-trade returns for Sharpe
        trade_returns = [t.pnl / self.initial_balance for t in trades]
        ret_arr = np.array(trade_returns) if trade_returns else np.array([0.0])
        sharpe = float(np.mean(ret_arr) / (np.std(ret_arr) + 1e-10))

        downside = ret_arr[ret_arr < 0]
        sortino = (
            float(np.mean(ret_arr) / (np.std(downside) + 1e-10))
            if len(downside) > 0
            else 0.0
        )

        avg_hours = (
            float(np.mean([t.holding_periods_hours for t in trades]))
            if trades
            else 0.0
        )

        final_eq = self.executor.equity
        total_return = (final_eq - self.initial_balance) / self.initial_balance * 100
        bars_effective = max(self._bars_traded, 1)
        # Annualized return based on time bars (hourly data: 8760 bars/year)
        annualized = ((final_eq / self.initial_balance) ** (8760 / bars_effective) - 1) * 100

        max_dd = self.risk.state.max_drawdown_pct * 100

        return BacktestResult(
            strategy="ppo_rl",
            symbol=self.settings.symbol,
            total_bars=total_bars,
            bars_traded=self._bars_traded,
            initial_balance=self.initial_balance,
            final_equity=round(final_eq, 2),
            total_return_pct=round(total_return, 4),
            annualized_return_pct=round(annualized, 4),
            max_drawdown_pct=round(max_dd, 4),
            sharpe_ratio=round(sharpe, 4),
            sortino_ratio=round(sortino, 4),
            win_rate=round(len(wins) / len(trades) * 100, 2) if trades else 0.0,
            profit_factor=round(gross_profit / gross_loss, 4),
            total_trades=len(trades),
            avg_trade_holding_hours=round(avg_hours, 2),
            trades=[
                {
                    "side": t.side.value,
                    "entry_price": t.entry_price,
                    "exit_price": t.exit_price,
                    "quantity": round(t.quantity, 4),
                    "pnl": round(t.pnl, 2),
                    "commission": round(t.commission, 2),
                    "holding_hours": round(t.holding_periods_hours, 2),
                    "reason": t.reason,
                }
                for t in trades
            ],
        )


def print_summary(result: BacktestResult) -> None:
    """Pretty-print backtest results."""
    print(f"\n{'='*55}")
    print(f"           BACKTEST RESULTS")
    print(f"{'='*55}")
    print(f"  Strategy:            {result.strategy}")
    print(f"  Symbol:              {result.symbol}")
    print(f"  Total bars:          {result.total_bars}")
    print(f"  Initial balance:     ${result.initial_balance:>10,.2f}")
    print(f"  Final equity:        ${result.final_equity:>10,.2f}")
    print(f"  Total return:        {result.total_return_pct:+>10.2f}%")
    print(f"  Annualized return:   {result.annualized_return_pct:+>10.2f}%")
    print(f"  Max drawdown:        {result.max_drawdown_pct:>10.2f}%")
    print(f"  Sharpe ratio:        {result.sharpe_ratio:>10.4f}")
    print(f"  Sortino ratio:       {result.sortino_ratio:>10.4f}")
    print(f"  Win rate:            {result.win_rate:>10.2f}%")
    print(f"  Profit factor:       {result.profit_factor:>10.4f}")
    print(f"  Total trades:        {result.total_trades:>10d}")
    print(f"  Avg holding (hrs):   {result.avg_trade_holding_hours:>10.2f}")
    print(f"{'='*55}")
    if result.trades:
        print(f"\n  Last 5 trades:")
        for t in result.trades[-5:]:
            print(
                f"    {t['side'].upper():4s}  Entry: {t['entry_price']:>10.2f}  "
                f"Exit: {t['exit_price']:>10.2f}  PnL: {t['pnl']:+>8.2f}"
            )
    print(f"{'='*55}\n")
