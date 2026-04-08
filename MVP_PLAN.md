# Autonomous AI Trading System: Concrete MVP Plan

**Objective:** Deliver an end-to-end, functional AI trading loop utilizing PyTorch models via `autoresearch` to power execution natively on testnet exchanges. The focus is strictly on minimum viability to prove the concept reliably before optimizing for production or scaling data/cloud costs.

## 🎯 MVP Core Definitions
A Minimum Viable Product for this fund proves one thing: **the model can output a signal, and the system can execute it without losing excess capital.**

**What is IN Scope:**
- 1-hour OHLCV data pipeline for BTC/USDT.
- Automated RL training loops capturing simple 12-feature states.
- Local FastAPI inference engine serving the latest `.pt` model.
- 14-days automated paper trading execution guarded strictly by hardcoded limits.

**What is OUT of Scope (Post-MVP phase):**
- Cloud deployment of the execution engine (Localhost is sufficient for MVP trial).
- Multi-asset strategy and portfolio reallocation (BTC/USDT ONLY).
- Advanced Order Book (LOB) data, macro indicators, and sentiment NLP (MiroFish).
- PostgreSQL or Timeseries deployments (SQLite is sufficient).

---

## 🗺️ Step-By-Step Implementation Roadmap

### Milestone 1: Data Plumbing & MVP Lab Setup
**Goal:** Seamless extraction of standard testing data for the autonomous engine.
1. **Data Download:** Pull 4 years of 1h historical BTC/USDT data via Binance Vision (free bulk zip).
2. **Feature Engineering:** Execute `feature_engine.py` to bake the 12 technical standard features into `features.parquet`.
3. **Environment Parity:** Init `.dvc` to track `features.parquet` and `labels.npy`. Apply `.devcontainer` standards ensuring consistency locally.

### Milestone 2: Automated Model Discovery Loop
**Goal:** Successful training runs producing weights `.pt` capable of outperforming baseline Hold.
1. **RL Training Execution:** Launch `train_quick.py` on cloud GPU with an active constraint on the environment: PPO algorithms maximizing the Sharpe ratio.
2. **Experiment Tracking:** W&B active. Agent learns strictly based on the 12 columns provided by MVP scope. Focus heavily on achieving model entropy stability.
3. **Save Artifact:** Push the winning `best_model.pt` via DVC and update MVP environment `.env`.

### Milestone 3: The Execution Bridge
**Goal:** Wrapping the raw PyTorch model efficiently into scalable signaling calls.
1. **Model API Service:** Run the `inference_api.py` serving `/predict` via FastAPI. It expects 12 engineered features and returns `{action: "BUY/SELL/HOLD", confidence: X%}`.
2. **Execution Hook:** Secure the `live_executor.py` logic around `ccxt` pulling latest tick feeds, assembling the OHLCV feature payload synchronously via `asyncio.to_thread()`, and querying the inference API continuously.

### Milestone 4: Paper Trading Quarantine (The Final MVP Gate)
**Goal:** 14 unbroken days of paper testing without breaking API connections or exhausting maximum drawdowns.
1. **Hardcoded Overrides:** Assert `-5% Stop-loss` and `+10% Take-Profit` limits into MVP code layer.
2. **Runtime Verification:** Push `.env` into `PAPER_TRADING=true`.
3. **Evaluate against Success Criteria:** 
    - Annualized Sharpe Ratio > 1.5
    - Daily drawdown remains stringently < 3%
    - 0 instances of consecutive error loops in websocket/REST handling.

---

> [!WARNING]
> Do NOT proceed to advanced milestones (MiroFish integration, Multi-Symbol scaling) until Milestone 4 satisfies all 3 success criteria continuously for 2 weeks.

## 🏁 Tech Stack Dependencies Used For MVP:
- **Python 3.11** (Strict environment boundaries)
- **PyTorch + Gymnasium** (Custom PPO Training Setup)
- **FastAPI** + **ccxt.pro** (Bridging Live Orders)
- **uv + DVC + W&B** (DevOps efficiency & logging)
