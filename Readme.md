\# Master Execution Plan: Autonomous AI Trading System



\## 📌 Project Overview

Building a fully automated trading infrastructure that utilizes Multi-Agent Orchestration (`agent-orchestrator`) for software engineering/DevOps, and Automated Machine Learning (`autoresearch`) for continuous Reinforcement Learning (RL) strategy discovery.



\## 🏗️ The Tech Stack

\### Domain A: The Brain (AI/ML Research Lab)

\* \*\*Core Framework:\*\* Python 3.11+, PyTorch.

\* \*\*RL Environment:\*\* `Gymnasium` (custom environment for OHLCV market data).

\* \*\*Automation:\*\* `karpathy/autoresearch` (modified for financial backtesting instead of LLM loss).

\* \*\*Data Source:\*\* `yfinance` or `ccxt` historical data downloaders.



\### Domain B: The Body (Execution \& Infrastructure)

\* \*\*Core Framework:\*\* Node.js (TypeScript) \& Python.

\* \*\*Orchestration:\*\* `ComposioHQ/agent-orchestrator` (managing coding agents).

\* \*\*Exchange API:\*\* `ccxt` (Unified API for Binance, Coinbase, etc.).

\* \*\*Database:\*\* PostgreSQL (with TimescaleDB extension for tick data) or SQLite for MVP.

\* \*\*Deployment:\*\* Docker \& Docker Compose.



\---



\## 🗺️ Phase 1: Foundation \& Data Plumbing

\*\*Goal:\*\* Set up the isolated environments and get historical data flowing.



\* \[ ] \*\*Step 1.1: Repo Setup\*\*

&#x20; \* Initialize monorepo with two main directories: `/research\_lab` (Python) and `/execution\_engine` (Node.js/Python).

\* \[ ] \*\*Step 1.2: Agent Orchestrator Initialization\*\*

&#x20; \* Deploy `agent-orchestrator` in the `/execution\_engine`.

&#x20; \* \*Prompt for Agent:\* "Create a secure `.env` template for exchange API keys. Do not hardcode any keys. Build a basic health-check script."

\* \[ ] \*\*Step 1.3: Data Ingestion Pipeline\*\*

&#x20; \* \*Task for Agent:\* Write a Python script using `ccxt` to download the last 4 years of 1-hour OHLCV data for BTC/USDT and ETH/USDT. Save this as a clean CSV/Parquet file in a `/data` folder.



\---



\## 🧠 Phase 2: The Research Lab (Adapting `autoresearch`)

\*\*Goal:\*\* Create an environment where AI can practice trading and modify its own brain.



\* \[ ] \*\*Step 2.1: Clone \& Modify `autoresearch`\*\*

&#x20; \* Clone the repo into `/research\_lab`.

\* \[ ] \*\*Step 2.2: Build the Gym Environment\*\*

&#x20; \* Create `trading\_env.py`. This script must load the CSV data from Phase 1. It gives the RL model three actions: `\[BUY, SELL, HOLD]`.

&#x20; \* Calculate reward based on Net Profit or Sharpe Ratio.

\* \[ ] \*\*Step 2.3: Modify the `train.py` target\*\*

&#x20; \* Rewrite `autoresearch`'s target goal. Instead of minimizing validation loss, instruct the LLM to maximize the reward function from `trading\_env.py`.

\* \[ ] \*\*Step 2.4: First Autonomous Run\*\*

&#x20; \* Run the orchestrator overnight. Goal: The system should generate at least 5 different PyTorch models (`.pt` files) that achieved positive returns on the training data.



\---



\## ⚙️ Phase 3: The Execution Engine (Building the Body)

\*\*Goal:\*\* Build the software that will connect the winning PyTorch models to a real exchange.



\* \[ ] \*\*Step 3.1: Live Data Websocket\*\*

&#x20; \* \*Task for Agent:\* Use `ccxt` pro or exchange-specific websockets to stream live price data into a local queue.

\* \[ ] \*\*Step 3.2: The Inference API\*\*

&#x20; \* \*Task for Agent:\* Wrap the best PyTorch model from Phase 2 into a lightweight FastAPI (Python) server. It should accept current market data, run inference, and return `BUY`, `SELL`, or `HOLD`.

\* \[ ] \*\*Step 3.3: Order Execution Module\*\*

&#x20; \* \*Task for Agent:\* Write the execution logic. If the Inference API says `BUY`, calculate position size (e.g., 2% of total balance), execute the market order, and log the transaction to SQLite/PostgreSQL.

\* \[ ] \*\*Step 3.4: Risk Management Hardcoding\*\*

&#x20; \* \*\*CRITICAL:\*\* Hardcode Stop-Loss (e.g., -5%) and Take-Profit (e.g., +10%) directly into the execution module. The RL model \*cannot\* override these safety nets.



\---



\## 🌉 Phase 4: Integration \& Paper Trading (The Proving Ground)

\*\*Goal:\*\* Connect the Brain to the Body in a safe, simulated environment.



\* \[ ] \*\*Step 4.1: CI/CD Pipeline Setup\*\*

&#x20; \* Configure GitHub Actions or a local bash script. When `autoresearch` finds a new "best model", it automatically pushes it to the Execution Engine's FastAPI server.

\* \[ ] \*\*Step 4.2: Enable Testnet / Paper Trading\*\*

&#x20; \* Configure `ccxt` to use the exchange's Testnet (simulated money). 

\* \[ ] \*\*Step 4.3: The 14-Day Quarantine\*\*

&#x20; \* Let the system run completely autonomously on the Testnet for 14 days. 

&#x20; \* Monitor the SQLite database logs. Are there API rate limit errors? Is the bot hallucinating trades? Fix bugs using `agent-orchestrator`.



\---



\## 🚀 Phase 5: Live Deployment

\*\*Goal:\*\* Go live with strict capital limits.



\* \[ ] \*\*Step 5.1: Capital Allocation\*\*

&#x20; \* Fund the live exchange account with \*only\* the absolute minimum required to trade (e.g., $50 - $100). 

\* \[ ] \*\*Step 5.2: Flip the Switch\*\*

&#x20; \* Change API keys from Testnet to Mainnet. 

\* \[ ] \*\*Step 5.3: Monitor \& Iterate\*\*

&#x20; \* Let the system run. Use `autoresearch` in the background to continuously retrain the model on the newest weekly data, pushing updates to the live server.



\## ⚠️ Absolute Rules for AI Agents

1\. \*\*Never\*\* expose `.env` files or API keys in code generation.

2\. \*\*Never\*\* execute a live trade without checking the available balance first.

3\. \*\*Always\*\* wrap API calls in `try/except` blocks to handle exchange timeouts or disconnects gracefully.

