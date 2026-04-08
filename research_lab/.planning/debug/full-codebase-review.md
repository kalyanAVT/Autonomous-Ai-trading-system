---
status: investigating
trigger: "Full codebase review across research_lab and execution_engine - check integration, imports, logic, and data flow"
created: 2026-04-05T00:00:00Z
updated: 2026-04-05T00:00:00Z
---

## Current Focus
hypothesis: Systematic review of all 15 files across both packages
test: Verify each check from the comprehensive review checklist
expecting: Identify all bugs, unused imports, integration mismatches, and logical issues
next_action: Analyze all findings and catalog issues

## Symptoms
expected: All files should have correct imports, matching column names, proper logic, no dead code
actual: Unknown - need to investigate
errors: N/A
reproduction: N/A - code review
started: First time review

## Eliminated

## Evidence

- timestamp: 2026-04-05T00:01:00Z
  checked: signal_generator.py FEATURE_COLUMNS vs feature_engine.py compute_all() output order
  found: FeatureEngine produces in order: log_ret_1, log_ret_3, log_ret_12, log_ret_24, roll_vol_12, roll_vol_24, roll_mean_12, roll_mean_24, momentum_14, vol_change_12, hl_spread, price_pos_24 — matches FEATURE_COLUMNS exactly (all 12 columns)
  implication: Column order is correct. No mismatch.

- timestamp: 2026-04-05T00:02:00Z
  checked: data_feed.py and backtester.py sys.path hack for research_lab imports
  found: data_feed.py uses sys.path.insert to import FeatureEngine (used at runtime). backtester.py does NOT import anything from research_lab — it loads pre-computed parquet. Both approaches are valid but different.
  implication: data_feed.py depends on research_lab being on disk. backtester.py has no runtime dependency on research_lab source (only on its output parquet/npy).

- timestamp: 2026-04-05T00:03:00Z
  checked: backtester.py data[FEATURE_COLUMNS] — parquet has indicator columns only
  found: data_preprocessor.py saves features.parquet with `features = self.df[self._feature_cols].copy()` — these are indicator columns only (no raw OHLCV). FEATURE_COLUMNS in signal_generator.py are indicator names.
  implication: Column names match. parquet has exactly the columns FEATURE_COLUMNS references.

- timestamp: 2026-04-05T00:04:00Z
  checked: paper_executor.py imports TradingEnv from research_lib trading_env
  found: Lines 28-31 — sys.path hack to import TradingEnv from research_lab/src/trading_env.py. BUT TradingEnv is NEVER referenced anywhere in paper_executor.py. Completely unused.
  implication: Dead import. Should be removed.

- timestamp: 2026-04-05T00:05:00Z
  checked: Unused imports across all 15 files
  found: 1) paper_executor.py: unused import of TradingEnv (confirmed). 2) config.py: `from __future__ import annotations` is unused (no type hints needing it yet, but harmless). 3) train_quick.py: torch imported inside main() — actually IS used on line 122 `torch.cuda.is_available()`.
  implication: #1 is the only true unused import to fix. #2 and #3 are fine.

- timestamp: 2026-04-05T00:06:00Z
  checked: Undefined variables across all 15 files
  found: No undefined variables found. All references resolved within scope.
  implication: No undefined variable issues.

- timestamp: 2026-04-05T00:07:00Z
  checked: Signal class location — is it in signal_generator.py, not models.py?
  found: Signal class is defined at line 37 of signal_generator.py as requested. models.py does NOT define Signal. Correct.
  implication: Signal placement is correct per specification.

- timestamp: 2026-04-05T00:08:00Z
  checked: MarketSnapshot defined in models.py and imported by anyone who needs it
  found: MarketSnapshot is defined at line 42 of models.py. Imported by: signal_generator.py (line 15), data_feed.py (line 20), backtester.py (line 19). All imports are `from .models import MarketSnapshot`.
  implication: MarketSnapshot placement and imports are correct.

- timestamp: 2026-04-05T00:09:00Z
  checked: paper_executor.py _apply_slippage is direction-aware
  found: Lines 224-230: `if side == Side.LONG: return price * (1.0 + slip)` (worse price for buy). `else: return price * (1.0 - slip)` (worse price for sell). Correct.
  implication: Slippage modeling is direction-aware and correct.

- timestamp: 2026-04-05T00:10:00Z
  checked: paper_executor.py _open_position balance math — can it go negative?
  found: Line 86: `self.balance -= exposure + commission`. Balance CAN go negative if exposure + commission > balance. E.g., balance=$50, exposure=$100 → balance becomes -50. However, the risk manager's check_pre_trade should catch excessive position sizing, and the risk manager checks consecutive losses and drawdown but does NOT check current balance vs required exposure.
  implication: BUG — No balance check in _open_position. If balance < exposure + commission, balance goes negative without any guard.

- timestamp: 2026-04-05T00:11:00Z
  checked: backtester.py pandas import is inside run()
  found: Line 79: `import pandas as pd` is inside the run() method. This is a lazy import, which works functionally. The rest of backtester.py uses only numpy for feature_matrix, which is fine.
  implication: Functional but unconventional. Not a bug.

- timestamp: 2026-04-05T00:12:00Z
  checked: risk_manager.py imports Signal from signal_generator.py — circular import?
  found: risk_manager.py line 18: `from .signal_generator import Signal`. Signal class depends only on numpy and MarketSnapshot — it does NOT import anything from .risk_manager or any module that would create a cycle.
  implication: No circular import. Import chain is clean.

- timestamp: 2026-04-05T00:13:00Z
  checked: main.py signal.signal() handlers and async event loop — does SIGINT properly wake the sleeping event loop?
  found: Line 141: Creates new event loop. Lines 146-147: Registers _shutdown handler. In _shutdown(): engine.stop() sets self._running=False. On SIGINT: Python calls _shutdown() setting _running=False, and run_until_complete() raises KeyboardInterrupt (Python 3.12+). KeyboardInterrupt caught at line 151, engine.stop() called again (idempotent), then run_until_complete called again. On second call, engine.run() checks _running, sees False, exits cleanly.
  implication: Works correctly. Minor: after KeyboardInterrupt catch, one more full loop iteration could occur before the while-loop check sees _running=False, but _shutdown handler fires synchronously with the signal so _running is already False by the time the next iteration checks.

- timestamp: 2026-04-05T00:14:00Z
  checked: research_lab/data_preprocessor.py saves features.parquet with FeatureEngine output columns
  found: Line 63: `features.to_parquet(str(output_path))` — saves the feature DataFrame with _feature_cols columns. Confirmed _feature_cols are the 12 indicator columns.
  implication: Parquet output contains exactly the 12 feature columns.

- timestamp: 2026-04-05T00:15:00Z
  checked: execution_engine/backtester.py loads parquet and selects FEATURE_COLUMNS
  found: Line 98: `feature_matrix = data[FEATURE_COLUMNS].values.astype(np.float32)`. Since parquet has exactly these 12 columns, this selection works.
  implication: Column selection is correct and will not KeyError.

- timestamp: 2026-04-05T00:16:00Z
  checked: Data flow full chain: research_lab -> parquet -> backtester -> inference
  found: Step 1: data_preprocessor.py saves features.parquet (indicator-only columns). Step 2: backtester.py reads parquet, selects FEATURE_COLUMNS. Step 3: signal_generator uses same FEATURE_COLUMNS to build inference input. All 12 columns match end-to-end.
  implication: Full data flow is correct. Column names and order are consistent across all consumers.

- timestamp: 2026-04-05T00:17:00Z
  checked: Syntax errors — all 15 files
  found: No syntax errors detected. All files are valid Python with correct indentation, matching brackets, valid decorators.
  implication: No syntax errors.

- timestamp: 2026-04-05T00:18:00Z
  checked: trading_env.py _apply_slippage (in research_lab, not execution_engine)
  found: Lines 101-104: `return price * (1.0 + slip)` for ALL trades regardless of direction. For SHORT positions selling, this should make price WORSE (subtract slip), but it ADDS slip instead, making the sale price HIGHER (better) for shorts.
  implication: BUG — training environment slippage is biased short-favorable. Agent trained on this will have incorrect slippage modeling for short positions.

- timestamp: 2026-04-05T00:19:00Z
  checked: run_backtest.py import paths — can `from src.backtester import Backtester` resolve?
  found: sys.path hack inserts parent.parent.parent which is the project root. Then `from src.backtester` looks for `<project_root>/src/backtester.py` but backtester is actually at `<project_root>/execution_engine/src/backtester.py`. Import FAILS. Same issue for `from src.config import Settings` and `from src.signal_generator import SignalGenerator`.
  implication: BUG — run_backtest.py will crash immediately on import. Needs relative imports to match the `python -m` docstring usage, or sys.path should be parent.parent (execution_engine/).

- timestamp: 2026-04-05T00:20:00Z
  checked: main.py — the KeyboardInterrupt re-runs engine.run() after stop()
  found: Line 153: `loop.run_until_complete(engine.run())` after KeyboardInterrupt. If _shutdown handler already set _running=False via SIGINT, this returns cleanly. If KeyboardInterrupt without SIGINT (Ctrl+C at terminal), _shutdown also sets _running=False. However, there IS one edge: if _shutdown is never called (possible on Windows), KeyboardInterrupt is caught, engine.stop() sets _running=False, then re-entering run_until_complete calls run() which checks _running=False at while, exits immediately. This is fine.
  implication: Works correctly but could be cleaner.

- timestamp: 2026-04-05T00:21:00Z
  checked: backtester.py — `self._bars_traded` incremented per bar regardless of trade
  found: Line 138: `self._bars_traded += 1` happens on EVERY iteration, not just when a trade occurs. The variable name implies "bars where trading occurred" but it's actually "total bars processed".
  implication: Naming is misleading but logic is fine — used for annualized return calculation where total bars is the right metric.

- timestamp: 2026-04-05T00:22:00Z
  checked: data_feed.py — `compute_features` takes the LAST row after `compute_all()` drops rows
  found: FeatureEngine.compute_all() drops index 0-500 (NaN rows from rolling). The `features_df.iloc[-1]` takes the most recent feature row. This is correct for the latest snapshot.
  implication: Correct behavior.

- timestamp: 2026-04-05T00:23:00Z
  checked: data_feed.py — `refresh` calls `exchange.fetch_ohlcv` which is blocking (not async)
  found: Line 130: `raw = self.exchange.fetch_ohlcv(...)` is a synchronous ccxt call inside an async method. This blocks the event loop. Same issue in fetch_history (line 52).
  implication: BUG — In the async main.py event loop, ccxt sync calls will block all other coroutines for the duration of the HTTP request.

## Resolution
root_cause: 3 bugs found and fixed, 1 remaining issue flagged for future work. See details in Evidence section.
fix: See files_changed for all modifications applied.
verification: Manual inspection confirms fixes. Full test coverage requires pytest with trained models (not available in this review context).
files_changed:
  - research_lab/src/trading_env.py: Direction-aware slippage (was all-biased-positive), fixed TradeRecord price ref
  - execution_engine/src/paper_executor.py: Removed dead TradingEnv import + sys.path hack; added balance guard in _open_position; added ValueError handler in execute_signal
  - execution_engine/src/run_backtest.py: Fixed sys.path resolution (was pointing to wrong directory); removed dead numpy/pandas imports; removed dead `symbol` variable
