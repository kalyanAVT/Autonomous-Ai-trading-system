"""Configuration for the execution engine."""

from __future__ import annotations

from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment / .env file."""

    # Exchange
    exchange_name: str = "binance"
    api_key: str = ""
    api_secret: str = ""
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"

    # Mode
    paper_trading: bool = True
    verbose: bool = False

    # Model
    model_path: str = ""

    # Risk limits
    max_position_pct: float = Field(default=0.1, ge=0.0, le=1.0)  # Max % of equity per position
    daily_loss_limit_pct: float = Field(default=0.03, ge=0.0, le=1.0)  # 3% daily loss limit
    max_drawdown_pct: float = Field(default=0.1, ge=0.0, le=1.0)  # 10% max drawdown before stop
    max_consecutive_losses: int = Field(default=3, ge=0)  # Stop after N consecutive losing trades
    commission_pct: float = Field(default=0.001, ge=0.0)  # 0.1%
    slippage_pct: float = Field(default=0.001, ge=0.0)  # 0.1%

    # Data feed
    poll_interval_seconds: int = Field(default=60, ge=10)  # How often to poll for new data
    feature_lookback: int = Field(default=500, ge=100)  # Candles to fetch for feature computation

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
