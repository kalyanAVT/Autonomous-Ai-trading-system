"""CLI entry point for running backtests.

Usage:
    python -m execution_engine.src.run_backtest \
        --model models/ppo_baseline.pt \
        --data ../research_lab/data/market_data.parquet \
        --prices ../research_lab/data/market_data.npy
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Ensure execution_engine's src package is importable when run as module:
#   python -m execution_engine.src.run_backtest
_parent = Path(__file__).resolve().parent.parent
if str(_parent) not in sys.path:
    sys.path.insert(0, str(_parent))

from src.analytics import generate_report
from src.backtester import Backtester, print_summary
from src.config import Settings
from src.signal_generator import SignalGenerator


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run PPO backtest on historical data")
    parser.add_argument(
        "--model", required=True, help="Path to trained PPO model (.pt file)"
    )
    parser.add_argument(
        "--data", required=True, help="Path to features parquet"
    )
    parser.add_argument(
        "--prices", default=None, help="Path to close prices .npy"
    )
    parser.add_argument("--output", default=None, help="Save results JSON path")
    parser.add_argument("--charts", default=None, help="Save charts to directory path")
    parser.add_argument("--initial-balance", type=float, default=10_000.0)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    settings = Settings()

    if args.prices:
        prices_path = args.prices
    else:
        # Default: same directory as data file with .npy extension
        prices_path = str(Path(args.data).with_suffix(".npy"))

    # Load model
    signal_gen = SignalGenerator(args.model)

    backtester = Backtester(
        signal_gen=signal_gen,
        settings=settings,
        initial_balance=args.initial_balance,
    )

    result = backtester.run(data_path=args.data, prices_path=prices_path)
    print_summary(result)

    if args.charts:
        chart_paths = generate_report(result, output_dir=args.charts)
        logging.info("Charts saved:")
        for name, path in chart_paths.items():
            logging.info("  %s: %s", name, path)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(
            {
                "strategy": result.strategy,
                "symbol": result.symbol,
                "total_bars": result.total_bars,
                "initial_balance": result.initial_balance,
                "final_equity": result.final_equity,
                "total_return_pct": result.total_return_pct,
                "annualized_return_pct": result.annualized_return_pct,
                "max_drawdown_pct": result.max_drawdown_pct,
                "sharpe_ratio": result.sharpe_ratio,
                "sortino_ratio": result.sortino_ratio,
                "win_rate": result.win_rate,
                "profit_factor": result.profit_factor,
                "total_trades": result.total_trades,
                "trades": result.trades,
            },
            indent=2,
        ))
        logging.info("Results saved to %s", output_path)


if __name__ == "__main__":
    main()
