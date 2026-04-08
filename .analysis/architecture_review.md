# Architecture Review — Autonomous AI Trading System

> Date: 2026-04-07
> Scope: Full review of `research_lab/` and `execution_engine/` modules

---

## 1. Current State Assessment

### What Exists
| Module | Files | Status |
|--------|-------|--------|
| **research_lab/src/** | `trading_env.py`, `feature_engine.py`, `data_fetcher.py`, `data_preprocessor.py`, `train_full.py`, `train_quick.py` | **Functional core** — RL environment + feature pipeline |
| **execution_engine/src/** | `main.py`, `config.py`, `data_feed.py`, `signal_generator.py`, `paper_executor.py`, `risk_manager.py`, `models.py`, `backtester.py`, `analytics.py`, `run_backtest.py` | **Near-complete** — live execution loop + backtesting |
| **config/** | `settings.py` | Minimal |

### Phase Coverage (from plan.txt)
| Phase | Goal | Status |
|-------|------|--------|
| Phase 1: Data Plumbing | ccxt data download pipeline | **70%** — `data_feed.py` exists, no standalone download script |
| Phase 2: Model Discovery | RL training with autoresearch | **40%** — `train_full.py`/`train_quick.py` exist but NOT integrated with karpathy/autoresearch. No `program.md` |
| Phase 3: Execution API | FastAPI server for model inference | **0%** — No FastAPI server exists |
| Phase 4: Paper Trading | 14-day testnet run | **60%** — `paper_executor.py` is complete, but needs `ccxt` testnet wiring |
| Phase 5: Live Deployment | Mainnet with limits | **30%** — Risk management exists, no live exchange wiring |

---

## 2. Architectural Gaps

### GAP-1: No Integration Between research_lab and execution_engine 【CRITICAL】
`execution_engine/src/data_feed.py:24-29` dynamically inserts `research_lab/src` into `sys.path` to import `FeatureEngine`. This is fragile:
- Breaks if either module is installed as a package
- No version pinning — feature column order drift between training/inference produces silent errors
- No test validating train/serve parity

**Fix**: Make `research_lab` a pip-installable package (`pyproject.toml`), add it as a dependency of `execution_engine`. Add a test that confirms feature columns in `signal_generator.py` match `feature_engine.py`.

### GAP-2: No FastAPI Inference Server (Phase 3) 【CRITICAL】
Plan Phase 3 requires a FastAPI server that accepts live data, runs inference, and returns BUY/SELL/HOLD. Currently, inference is embedded in `main.py`'s synchronous loop.

**Impact**: Cannot support:
- Multiple concurrent strategies
- HTTP-based monitoring or external orchestration
- The "Execution API" described in the plan

**Fix**: Create `execution_engine/src/server.py` with:
```python
POST /v1/signal  -> accepts OHLCV features, returns {action, confidence, side}
GET  /v1/health  -> engine status
GET  /v1/summary -> portfolio state
```

### GAP-3: No Karpathy autoresearch Integration (Phase 2) 【HIGH】
The plan references `karpathy/autoresearch` as the core research engine, but:
- The repo is NOT cloned as a submodule
- No `program.md` file exists (research objective is not configured)
- `train_full.py` and `train_quick.py` are standalone scripts with no autoresearch integration

**Fix**: Either:
1. Clone `autoresearch` and create `program.md` with the trading objective, OR
2. The existing `train_quick.py`/`train_full.py` scripts implement their own search loop — document which approach to use

### GAP-4: No Database Layer 【HIGH】
Plan states: "SQLite / PostgreSQL: Logs every trade, simulated or real." No database exists. Trade history is held in-memory (`PaperExecutor.trade_history: list[TradeRecord]`). On restart, all history is lost.

**Fix**: Add `execution_engine/src/db.py` with SQLite for:
- Trade logging (entry, exit, PnL, reason)
- Session tracking (start/end equity, duration)
- Model version metadata (which .pt file produced which trades)

### GAP-5: No Backtest-to-Live Bridge 【MEDIUM】
`backtester.py` and `run_backtest.py` can run backtests on pre-computed parquet, but there's no workflow that:
1. Takes a winning model from backtest
2. Promotes it to paper trading
3. Promotes paper → live

**Fix**: Add model promotion workflow with metadata (`model_name`, `backtest_sharpe`, `paper_days`, `go_live_date`).

---

## 3. Training vs Inference Feature Mismatch Risk

`signal_generator.py:21-33` hardcodes:
```python
FEATURE_COLUMNS = [
    "log_ret_1", "log_ret_3", "log_ret_12", "log_ret_24",
    "roll_vol_12", "roll_vol_24", "roll_mean_12", "roll_mean_24",
    "momentum_14", "vol_change_12", "hl_spread", "price_pos_24",
]  # 12 columns
```

`feature_engine.py` computes the same columns via `_add_*` methods but the order depends on the **call order in `compute_all()`** (lines 107-128):
1. log_ret_1, log_ret_3, log_ret_12, log_ret_24 ✓
2. roll_vol_12, roll_vol_24 ✓
3. roll_mean_12, roll_mean_24 ✓
4. momentum_14 ✓
5. vol_change_12 ✓
6. hl_spread ✓
7. price_pos_24 ✓

**Current alignment: CORRECT.** This is fragile — anyone adding a feature to `compute_all()` without updating `FEATURE_COLUMNS` will produce silently wrong signals.

**Fix**: Export `FEATURE_COLUMNS` from `feature_engine.py` and have `signal_generator.py` import it from there. Single source of truth.

---

## 4. Missing Components Summary

| Component | Priority | Phase | Description |
|-----------|----------|-------|-------------|
| FastAPI Server | CRITICAL | 3 | HTTP API for model inference |
| Database (SQLite) | CRITICAL | 4 | Persistent trade/session logging |
| autoresearch Integration | CRITICAL | 2 | ML model discovery pipeline |
| ccxt Testnet Wiring | HIGH | 4 | Paper trading with testnet exchange |
| Live Exchange Wiring | HIGH | 5 | Mainnet execution with API keys |
| .env Template | HIGH | 1 | Secure credential management |
| Model Promotion Workflow | MEDIUM | 5 | Backtest → Paper → Live gates |
| Feature Column Single Source | MEDIUM | 2 | Prevent training/inference drift |
| Docker/Containerization | LOW | 5 | Reproducible deployment |

---

## 5. Recommendations

### Immediate (Phase 1-2)
1. Add `pyproject.toml` to `research_lab` to make it an installable package — `execution_engine` should `pip install -e ../research_lab` instead of `sys.path.insert`
2. Move `FEATURE_COLUMNS` to `feature_engine.py` — single source of truth
3. Create `.env.example` with all required variables
4. Add a standalone data download script using ccxt

### Short Term (Phase 3-4)
1. Build FastAPI inference server
2. Add SQLite database layer for trade persistence
3. Wire `ccxt` testnet for paper trading
4. Write integration tests: feature pipeline → model → paper executor

### Long Term (Phase 5)
1. Model promotion workflow with approval gates
2. Live exchange execution with failover
3. Monitoring/alerting dashboard
4. Containerization for deployment
