# Roadmap

## Phase 1: Data & Research Lab [COMPLETE]
- Download 4+ yr BTC/USDT 1h OHLCV via ccxt → data/btc_1h.parquet
- Feature engine: 12 indicator columns → features.parquet + labels.npy
- Custom Gymnasium env (trading_env.py) wraps parquet for RL training
- autoresearch modified to maximize Sharpe Ratio (not LM loss)
- Bugs fixed: direction-aware slippage, balance guard, sys.path, dead imports

## Phase 1.5: Infrastructure Hardening [IN PROGRESS]
- Fix data_feed.py async bug
- Set up W&B experiment tracking
- Set up DVC
- Set up Dev Container
- Set up pre-commit hooks
- Download Binance Vision bulk data
- Download LOB (Limit Order Book) data
- Download G-Research Kaggle dataset

## Phase 2: Model Training (Cloud GPU) [PENDING]
- Complete Phase 1.5 infrastructure before training
- Upload research_lab/ + data/ to Colab or RunPod
- Verify W&B init is in train_quick.py before running
- Run: uv run python src/train_quick.py (5 min cap per run)
- Let autoresearch iterate 20+ runs overnight
- Monitor W&B dashboard
- Download best_model.pt → local data/ folder
- dvc add data/best_model.pt && git commit
- Update .env: MODEL_PATH=data/best_model.pt
- Validate: python src/run_backtest.py --model data/best_model.pt
- ACCEPT if: Sharpe > 1.5, max drawdown < 15%, win rate > 50%
- REJECT if: agent only learned HOLD (entropy too low), or fits only bull market

## Phase 3: Execution Engine [PENDING]
- Fix async data_feed.py bug (MUST be done first)
- Build inference_api.py — FastAPI server wrapping best_model.pt
- Connect ccxt.pro testnet credentials in .env
- End-to-end integration test
- Verify hardcoded safety limits are active in risk_manager.py

## Phase 4: Paper Trading 14-Day Quarantine [PENDING]
- Set PAPER_TRADING=true in .env
- Run for 14 consecutive days on Binance Testnet
- Open dashboard/index.html daily
- PASS CRITERIA: Sharpe > 1.5 annualized, daily DD < 3%, zero crash loops
- Tune risk_manager.py thresholds based on observed behavior
- Log issues to CHECKLIST.md. Fix using agent-orchestrator

## Phase 5: Live Deployment [PENDING]
- Change EXCHANGE_API_KEY + EXCHANGE_SECRET from Testnet to Mainnet in .env
- Set PAPER_TRADING=false
- Fund exchange account with $50-100 MAXIMUM
- Monitor first 48 hours continuously via dashboard
- autoresearch continues retraining weekly on new data in background
- agent-orchestrator handles code fixes autonomously via PRs
- After 30 days profitable: consider scaling capital (still max 2% per trade)

**Requirements:**
- EX-01: Fix data_feed.py async bug (current phase)