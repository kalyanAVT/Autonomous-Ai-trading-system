"""Configuration for the execution engine."""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file.

    Env var aliases map to the names used in .env.example/.env
    so the same .env works for both paper and live trading.
    """

    # Exchange
    exchange_name: str = Field(default="binance", alias="EXCHANGE_NAME")
    api_key: str = Field(default="", alias="BINANCE_API_KEY")
    api_secret: str = Field(default="", alias="BINANCE_API_SECRET")
    symbol: str = Field(default="BTC/USDT", alias="TRADING_SYMBOLS")
    timeframe: str = Field(default="1h", alias="TRADING_TIMEFRAME")

    # Mode
    paper_trading: bool = Field(default=True, alias="PAPER_TRADING")
    use_testnet: bool = Field(default=True, alias="USE_TESTNET")
    verbose: bool = False

    # Model
    model_path: str = Field(default="", alias="MODEL_PATH")

    # Risk limits
    max_position_pct: float = Field(default=0.1, alias="MAX_POSITION_SIZE_PCT")
    daily_loss_limit_pct: float = Field(default=0.03, alias="MAX_DAILY_LOSS_PCT")
    max_drawdown_pct: float = Field(default=0.1, alias="MAX_DRAWDOWN_PCT")
    max_consecutive_losses: int = Field(default=3, alias="MAX_CONSECUTIVE_LOSSES")
    commission_pct: float = Field(default=0.001)
    slippage_pct: float = Field(default=0.001)

    # Stop-loss / Take-profit (hardcoded safety limits per plan rules)
    stop_loss_pct: float = Field(default=0.05, alias="STOP_LOSS_PCT")
    take_profit_pct: float = Field(default=0.10, alias="TAKE_PROFIT_PCT")

    # Data feed
    poll_interval_seconds: int = Field(default=60)
    feature_lookback: int = Field(default=500, alias="FEATURE_LOOKBACK")

    # Social intelligence
    cryptopanic_api: str = Field(default="", alias="CRYPTOPANIC_API")

    # On-chain / derivatives
    symbol_onchain: str = Field(default="BTC", alias="ONCHAIN_SYMBOL")
    coingecko_id: str = Field(default="bitcoin", alias="COINGECKO_ID")

    # Agent orchestration
    use_agent_consensus: bool = Field(default=True)
    use_direct_model: bool = Field(default=True)
    agent_fusion_weight: float = Field(default=0.6)
    model_fusion_weight: float = Field(default=0.4)
    consensus_threshold: float = Field(default=0.1)
    min_confidence_threshold: float = Field(default=0.15)
    social_poll_interval: int = Field(default=300)

    model_config = {"populate_by_name": True, "env_file": ".env", "env_file_encoding": "utf-8"}
