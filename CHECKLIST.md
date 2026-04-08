# Project Checklist — Autonomous AI Trading System

> Created: 2026-04-08 | Last Updated: 2026-04-08
> Use this checklist to track completion status across sessions.
> Mark items as `[x]` when completed, `[~]` when in-progress.

---

## Phase 1: Research Lab (Model Training)

### Data Pipeline
- [x] Data fetcher via ccxt/Binance (`data_fetcher.py`) — fetches hourly OHLCV
- [x] Feature engineering pipeline (`feature_engine.py`) — RSI, log returns, vol, etc.
- [x] Gymnasium trading environment (`trading_env.py`) — reward = Sharpe - drawdown - costs
- [x] Quick PPO training script (`train_quick.py`) — single split training
- [x] Walk-forward PPO training script (`train_full.py`) — rolling window training
- [x] **Data preparation script** (`execution_engine/src/download_data.py`) — one CLI: download → feature compute → parquet + .npy
- [x] pyproject.toml + uv lock — dependencies managed
- [x] Data collected through 2026-02-09 (34,955 candles = ~4.3 years)

### Model Training (Cloud)
- [ ] Trained PPO model artifact (`.pt` or `.zip`) — needs GPU/cloud training
- [ ] Model validation metrics recorded — Sharpe, returns, max drawdown
- [ ] TensorBoard / W&B logs from training run

### Research Lab Gaps
- [ ] Autoresearch / hyperparameter optimization (evolutionary search over reward weights, env params)
- [ ] Multi-symbol support (ETH, SOL, etc.)
- [ ] Feature ablation / importance analysis

---

## Phase 2: Execution Engine (Paper Trading)

### Core Infrastructure
- [x] Execution engine orchestration (`main.py`) — data → fusion → execution loop
- [x] Configuration via pydantic + .env (`config.py`)
- [x] MarketSnapshot + PPO inference (`signal_generator.py`) — loads `.pt` and predicts
- [x] Signal fusion engine (`signal_fusion.py`) — agent + model weighted fusion
- [x] Data feed with candle buffer (`data_feed.py`) — ccxt polling + feature parity
- [x] Risk manager (`risk_manager.py`) — pre-trade checks, daily limits, drawdown
- [x] Paper executor (`paper_executor.py`) — simulated orders with slippage/commission
- [x] Multi-agent framework (`agents.py`) — Technical, Sentiment, OnChain, Risk agents
- [x] Social intelligence (`social_intel.py`) — Reddit, news, Fear & Greed, market regime
- [x] Database persistence (`db.py`) — SQLite session + trade logging
- [x] FastAPI server (`server.py`) — HTTP interface for signals
- [x] pyproject.toml + uv — dependencies managed

### Paper Trading Readiness
- [ ] `.env` configured with `MODEL_PATH` pointing to trained model
- [ ] Paper trading run for 1 week (target: 5%+ Sharpe, <3% drawdown daily)
- [ ] Performance metrics logged and reviewed
- [ ] Risk manager thresholds tuned based on paper results

### Verified (Code Review Complete)
- [x] RiskManager — pre-trade checks, circuit breaker, daily limits, drawdown tracking all solid
- [x] Models module — Order, Position, TradeRecord, RiskState types complete
- [x] SocialIntelligence — Fear & Greed, Reddit, CoinGecko, CryptoPanic (all free tiers, graceful fallbacks)
- [x] OnChainIntelligence — Funding rates (Binance), Liquidations, Exchange flows (CoinGecko)
- [x] LiveExecutor — real ccxt execution, mirrors PaperExecutor API, sandbox mode

---

## Phase 3: Live Trading

### Live Execution
- [x] LiveExecutor implementation (`live_executor.py`) — real Binance orders, mirrors PaperExecutor API
- [ ] Testnet mode testing (`use_testnet=true`)
- [ ] Order status tracking + retry logic
- [ ] WebSocket data feed (optional, replaces polling)
- [ ] Position management — trailing stops, scaling in/out

### Risk & Compliance
- [ ] Position sizing validated against account balance
- [ ] Kill switch — manual + automated (max loss, max drawdown, disconnect)
- [ ] Trade journal export — CSV/P&L reporting
- [ ] Paper vs live performance comparison

---

## Phase 4: Model Iteration

### Feedback Loop
- [ ] Trade outcome feedback to model retraining
- [ ] Agent performance tracking — accuracy per agent
- [ ] Fusion weight adaptation based on regime
- [ ] Periodic model retraining pipeline (weekly/monthly)

### Extensions
- [ ] Multi-timeframe signals (15m entries + 1h trend filter)
- [ ] Portfolio diversification (multiple symbols simultaneously)
- [ ] News/Twitter sentiment integration (if API costs justified)
- [ ] On-chain data from alternative providers (Glassnode, Dune)

---

## Files Summary

### Research Lab — Key Files
| File | Status |
|------|--------|
| `research_lab/src/feature_engine.py` | Complete |
| `research_lab/src/trading_env.py` | Complete |
| `research_lab/src/data_fetcher.py` | Complete |
| `research_lab/src/train_quick.py` | Complete |
| `research_lab/src/train_full.py` | Complete |

### Execution Engine — Key Files
| File | Status |
|------|--------|
| `execution_engine/src/main.py` | Complete |
| `execution_engine/src/config.py` | Complete |
| `execution_engine/src/models.py` | Complete |
| `execution_engine/src/data_feed.py` | Complete |
| `execution_engine/src/signal_generator.py` | Complete |
| `execution_engine/src/signal_fusion.py` | Complete |
| `execution_engine/src/agents.py` | Complete |
| `execution_engine/src/social_intel.py` | Complete |
| `execution_engine/src/paper_executor.py` | Complete |
| `execution_engine/src/live_executor.py` | Complete — reviewed |
| `execution_engine/src/risk_manager.py` | Complete — reviewed |
| `execution_engine/src/db.py` | Complete |
| `execution_engine/src/server.py` | Complete |

---

## Next Session Priorities
1. Build the data preparation script (download + feature compute → parquet + .npy)
2. Train PPO model in cloud (Google Colab / RunPod)
3. Review risk_manager.py and live_executor.py implementations
4. End-to-end paper trading test with a model
5. Fix any runtime errors before live deployment
