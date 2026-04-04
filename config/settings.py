"""
Shared configuration loaded from environment variables.
Used by both research_lab and execution_engine.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
_project_root = Path(__file__).resolve().parent.parent
env_path = _project_root / ".env"
load_dotenv(env_path)


# --- Binance ---
BINANCE_API_KEY = os.environ.get("BINANCE_API_KEY", "")
BINANCE_API_SECRET = os.environ.get("BINANCE_API_SECRET", "")
USE_TESTNET = os.environ.get("USE_TESTNET", "true").lower() == "true"

if not USE_TESTNET and not BINANCE_API_KEY:
    raise ValueError("BINANCE_API_KEY is required for mainnet trading")

# --- Trading ---
TRADING_SYMBOLS = os.environ.get("TRADING_SYMBOLS", "BTC/USDT").split(",")
TRADING_TIMEFRAME = os.environ.get("TRADING_TIMEFRAME", "1h")

# --- Risk Management ---
MAX_POSITION_SIZE_PCT = float(os.environ.get("MAX_POSITION_SIZE_PCT", "0.05"))
STOP_LOSS_PCT = float(os.environ.get("STOP_LOSS_PCT", "0.05"))
TAKE_PROFIT_PCT = float(os.environ.get("TAKE_PROFIT_PCT", "0.10"))
MAX_DAILY_LOSS_PCT = float(os.environ.get("MAX_DAILY_LOSS_PCT", "0.02"))
MAX_DRAWDOWN_PCT = float(os.environ.get("MAX_DRAWDOWN_PCT", "0.10"))
COOLDOWN_MINUTES = int(os.environ.get("COOLDOWN_MINUTES", "60"))

# --- ML ---
WANDB_PROJECT = os.environ.get("WANDB_PROJECT", "auto-trading")
WANDB_API_KEY = os.environ.get("WANDB_API_KEY", "")
AUTORESEARCH_ITERATIONS = int(os.environ.get("AUTORESEARCH_ITERATIONS", "100"))
MAX_TRAIN_MINUTES = int(os.environ.get("MAX_TRAIN_MINUTES", "5"))

# --- Logging ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO")
LOG_FILE = _project_root / os.environ.get("LOG_FILE", "logs/trading.log")

# --- Execution Engine ---
DATABASE_URL = os.environ.get("DATABASE_URL", "sqlite:///data/trades.db")
API_HOST = os.environ.get("API_HOST", "0.0.0.0")
API_PORT = int(os.environ.get("API_PORT", "8000"))
