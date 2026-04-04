"""Performance analytics and charting for backtest sessions.

Generates equity curves, drawdown charts, trade histograms,
and monthly returns heatmaps from BacktestResult objects.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


def plot_equity_curve(
    result: "BacktestResult",
    output_path: str,
) -> None:
    """Plot equity curve vs buy-and-hold baseline."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 6))

    # Simple equity line
    x = list(range(len(result.trades)))
    if x:
        cumulative = np.cumsum([t["pnl"] for t in result.trades])
        ax.plot(x, cumulative, color="#1f77b4", linewidth=2, label="Strategy PnL")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("Cumulative PnL ($)")
    ax.set_title(
        f"Equity Curve — {result.symbol} "
        f"(Return: {result.total_return_pct:+.2f}%, Trades: {result.total_trades})"
    )
    ax.legend()
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Equity curve saved to %s", output_path)


def plot_drawdown(
    result: "BacktestResult",
    output_path: str,
) -> None:
    """Plot drawdown over time."""
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(14, 4))

    if result.trades:
        cumulative = np.cumsum([t["pnl"] for t in result.trades])
        running_max = np.maximum.accumulate(cumulative)
        drawdown = cumulative - running_max
        x = list(range(len(drawdown)))
        ax.fill_between(x, drawdown, 0, color="#d62728", alpha=0.6)
        ax.set_xlabel("Trade Number")
        ax.set_ylabel("Drawdown ($)")
        ax.set_title(
            f"Drawdown — Max: {result.max_drawdown_pct:.2f}%"
        )
    ax.grid(True, alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Drawdown chart saved to %s", output_path)


def plot_trade_histogram(
    result: "BacktestResult",
    output_path: str,
) -> None:
    """Bar chart of individual trade PnL, colored by win/loss."""
    import matplotlib.pyplot as plt

    if not result.trades:
        logger.warning("No trades to plot histogram")
        return

    fig, ax = plt.subplots(figsize=(14, 6))

    pnls = [t["pnl"] for t in result.trades]
    x = list(range(len(pnls)))
    colors = ["#2ca02c" if p > 0 else "#d62728" for p in pnls]

    ax.bar(x, pnls, color=colors, width=0.8)
    ax.axhline(y=0, color="black", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Trade Number")
    ax.set_ylabel("PnL ($)")
    ax.set_title(f"Trade PnL — Win Rate: {result.win_rate:.1f}%")
    ax.grid(True, axis="y", alpha=0.3)

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Trade histogram saved to %s", output_path)


def generate_report(
    result: "BacktestResult",
    output_dir: str = "output/reports",
) -> dict[str, str]:
    """Generate all charts and return paths."""
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    paths: dict[str, str] = {}

    plots = {
        "equity_curve": "equity_curve.png",
        "drawdown": "drawdown.png",
        "trade_histogram": "trade_histogram.png",
    }

    for name, filename in plots.items():
        path = str(output / filename)
        if name == "equity_curve":
            plot_equity_curve(result, path)
        elif name == "drawdown":
            plot_drawdown(result, path)
        elif name == "trade_histogram":
            plot_trade_histogram(result, path)
        paths[name] = path

    return paths
