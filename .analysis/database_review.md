# Database & Persistence Review — Autonomous AI Trading System

> Date: 2026-04-07
> Scope: Data storage, trade logging, model tracking, audit trail

---

## 1. Current State

**No persistent database exists.** All trade and portfolio data is held in memory:

| Data Type | Current Storage | Persistence |
|-----------|----------------|-------------|
| Trade records | `PaperExecutor.trade_history: list[TradeRecord]` | Lost on restart |
| Order history | `PaperExecutor.order_history: list[Order]` | Lost on restart |
| Risk state | `RiskManager._state: RiskState` | Lost on restart |
| Model files | `.pt` files on disk (via `model_path` setting) | Persisted (file system) |
| Candle buffer | `DataFeed._candle_buffer: list[OHLCVCandle]` | Lost on restart |
| Backtest results | `BacktestResult` objects | Only saved via `analytics.py` charts (PNG) |

---

## 2. Missing Persistence Layer

### MUST-HAVE: Trade Database
The plan states: "SQLite / PostgreSQL: Logs every trade, simulated or real, to track win rates and system health."

**Recommended Schema**:
```sql
CREATE TABLE trades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    session_id TEXT NOT NULL,
    symbol TEXT NOT NULL,
    side TEXT NOT NULL,
    entry_price REAL NOT NULL,
    exit_price REAL NOT NULL,
    quantity REAL NOT NULL,
    pnl REAL NOT NULL,
    commission REAL NOT NULL,
    stop_loss REAL,
    take_profit REAL,
    reason TEXT NOT NULL,
    model_version TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    start_time TEXT NOT NULL,
    end_time TEXT,
    initial_balance REAL NOT NULL,
    final_equity REAL,
    total_trades INTEGER DEFAULT 0,
    mode TEXT NOT NULL,  -- 'paper' or 'live'
    model_path TEXT
);

CREATE TABLE models (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    path TEXT NOT NULL,
    training_date TEXT NOT NULL,
    backtest_sharpe REAL,
    backtest_return_pct REAL,
    paper_days INTEGER DEFAULT 0,
    promoted_to_live BOOLEAN DEFAULT 0
);
```

**Impact of Missing DB**:
- Cannot survive process restarts
- Cannot review historical performance across sessions
- Cannot audit which model produced which trades
- No data for regulatory/compliance review (Phase 5)

---

## 3. Feature Parity Risk

### Train/Serve Feature Drift
`research_lab/src/feature_engine.py` → `execution_engine/src/data_feed.py`

Both compute features using the same `FeatureEngine` class (because `data_feed.py` imports it via `sys.path` hack). However:

1. **No validation at runtime** that the model was trained on the same feature version
2. **No feature version metadata** — if a new column is added to `compute_all()`, old models silently get wrong inputs
3. **No feature hash/signature** stored with the model for integrity checking

**Recommendation**: Store a `feature_signature` (hash of column names + their order + normalization parameters) alongside each model. Validate at load time.

---

## 4. Candle Buffer Efficiency

`data_feed.py:159-168`: Maintains an in-memory buffer of ~700 OHLCV candles.

**Current approach**: Poll-based, appending new candles, trimming old ones.

**For Phase 4-5 (live trading)**, this should be backed by a time-series store so:
- Historical candles survive restarts
- Can backfill missing candles on reconnection
- Can query "what happened at this timestamp?" for audit

---

## 5. Model Versioning

Currently: `config.py:24` — `model_path: str = ""`

There is no model registry or versioning. The system loads whatever `.pt` file is pointed at.

**Recommended**: Add a model registry table that tracks:
- File path and SHA256 hash
- Training parameters
- Backtest metrics
- Promotion status (backtest/paper/live)
- Replacement history

---

## 6. Recommendations by Phase

### Phase 1-2 (Now)
- [ ] Add `db.py` with SQLite backend
- [ ] Create `TradeLogger` wrapper that writes to both DB and in-memory
- [ ] Store `session_id` with every trade

### Phase 3
- [ ] Add model registry for tracking discovered models
- [ ] Store feature signatures with models for integrity validation

### Phase 4
- [ ] Paper trading session persistence — survive restarts
- [ ] Query endpoint for trade analytics (win rate by day, avg holding, etc.)

### Phase 5
- [ ] Audit trail: no trade should be modifiable or deletable after execution
- [ ] Consider PostgreSQL for multi-reader access (monitoring dashboards)
- [ ] Backfill capability for missed candles during downtime
