"""Social intelligence module — free data sources for trading sentiment.

Collects sentiment from:
- Reddit (r/CryptoCurrency, r/Bitcoin)
- CryptoPanic news aggregation
- Google Trends
- Alternative.me Fear & Greed Index
- CoinGecko market data

All sources have free tiers requiring no API keys (or free keys).
"""

from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Optional

import aiohttp

logger = logging.getLogger(__name__)


class SocialIntelligence:
    """Aggregates free social/market sentiment sources."""

    FEAR_GREED_URL = "https://api.alternative.me/fng/?limit=1"
    GOOGLE_TRENDS_BASE = "https://trends.google.com"
    REDDIT_URL = "https://www.reddit.com/r/{sub}/hot.json?limit=25"
    CRYPTOPANIC_URL = "https://cryptopanic.com/api/v1/posts/?currencies=BTC,ETH"

    def __init__(self, cryptopanic_api: str = "", reddit_user_agent: str = "AI-Trading-Bot/1.0"):
        self.cryptopanic_api = cryptopanic_api
        self.reddit_user_agent = reddit_user_agent
        self._cache: dict = {}
        self._cache_time: dict = {}

    async def collect_all(self) -> dict:
        """Fetch all available sentiment sources concurrently."""
        results = {}
        tasks = [
            self._fetch_fear_greed(),
            self._fetch_reddit_sentiment("CryptoCurrency"),
            self._fetch_reddit_sentiment("Bitcoin"),
            self._fetch_google_trends(),
        ]
        if self.cryptopanic_api:
            tasks.append(self._fetch_cryptopanic())
        else:
            tasks.append(self._fetch_coingecko_sentiment())

        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for result in gathered:
            if isinstance(result, dict):
                results.update(result)
            elif isinstance(result, Exception):
                logger.warning("Sentiment source error: %s", result)

        logger.info("Social intelligence collected: %s", results)
        return results

    async def _fetch_fear_greed(self) -> dict:
        """Alternative.me Fear & Greed Index (free, no key)."""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.FEAR_GREED_URL, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    data = await resp.json()
                    value = int(data["data"][0]["value"])
                    label = data["data"][0]["value_classification"]
                    logger.info("Fear & Greed: %s (%d)", label, value)
                    return {"fear_greed": value, "fear_greed_label": label}
        except Exception as e:
            logger.warning("Failed to fetch Fear & Greed: %s", e)
            return {"fear_greed": 50, "fear_greed_label": "neutral"}

    async def _fetch_reddit_sentiment(self, subreddit: str) -> dict:
        """Scrape Reddit hot posts for crypto sentiment using simple keyword scoring."""
        try:
            headers = {"User-Agent": self.reddit_user_agent}
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    self.REDDIT_URL.format(sub=subreddit),
                    headers=headers,
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
                    posts = data.get("data", {}).get("children", [])
                    text = " ".join(
                        p.get("data", {}).get("title", "").lower()
                        for p in posts
                        if p.get("data", {}).get("title")
                    )
                    # Simple sentiment scoring
                    score = self._simple_sentiment(text)
                    logger.info("Reddit r/%s sentiment: %.3f", subreddit, score)
                    return {f"reddit_{subreddit.lower()}_sentiment": score}
        except Exception as e:
            logger.warning("Failed to fetch Reddit r/%s: %s", subreddit, e)
            return {f"reddit_{subreddit.lower()}_sentiment": 0.0}

    async def _fetch_cryptopanic(self) -> dict:
        """CryptoPanic news aggregation (free tier)."""
        try:
            url = f"{self.CRYPTOPANIC_URL}&auth_token={self.cryptopanic_api}"
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    data = await resp.json()
                    results = data.get("results", [])[:50]
                    # Sentiment from CryptoPanic's own labels
                    bullish = sum(1 for r in results if r.get("sentiment") == "positive")
                    bearish = sum(1 for r in results if r.get("sentiment") == "negative")
                    total = max(len(results), 1)
                    score = (bullish - bearish) / total
                    logger.info(
                        "CryptoPanic: %d bullish, %d bearish, score=%.3f",
                        bullish,
                        bearish,
                        score,
                    )
                    return {"news_sentiment": score}
        except Exception as e:
            logger.warning("Failed to fetch CryptoPanic: %s", e)
            return {"news_sentiment": 0.0}

    async def _fetch_coingecko_sentiment(self) -> dict:
        """Fallback: use CoinGecko market data as sentiment proxy.

        Uses:
        - Price change 24h (momentum)
        - Market cap dominance shift
        """
        try:
            async with aiohttp.ClientSession() as session:
                # BTC price and stats
                async with session.get(
                    "https://api.coingecko.com/api/v3/coins/bitcoin?localization=false&tickers=false"
                    "&community_data=false&developer_data=false&sparkline=false",
                    timeout=aiohttp.ClientTimeout(total=15),
                ) as resp:
                    data = await resp.json()
                    price_change_24h = data.get("market_data", {}).get(
                        "price_change_percentage_24h", 0
                    )
                    # Normalize: +5% = 1.0, -5% = -1.0
                    score = max(-1.0, min(1.0, price_change_24h / 5.0))
                    logger.info("CoinGecko BTC 24h change: %.2f%% → score=%.3f", price_change_24h, score)
                    return {"news_sentiment": score}
        except Exception as e:
            logger.warning("Failed to fetch CoinGecko: %s", e)
            return {"news_sentiment": 0.0}

    async def _fetch_google_trends(self) -> dict:
        """Google Trends proxy using search volume estimation.

        Since Google Trends doesn't have a public API, we use a heuristic
        based on CoinGecko volume data as a proxy for public interest.
        High volume spike = high public interest.
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://api.coingecko.com/api/v3/global"
                async with session.get(
                    url, timeout=aiohttp.ClientTimeout(total=15)
                ) as resp:
                    data = await resp.json()
                    market_data = data.get("data", {})
                    volume_change = market_data.get(
                        "total_volume", {}
                    ).get("usd_percentage_change_24h", 0)

                    # Volume increases = more public interest
                    # Normalize: +50% volume = 1.0
                    score = max(-1.0, min(1.0, volume_change / 50.0))
                    logger.info("Google Trends proxy (volume): %.3f", score)
                    return {"google_trends": score}
        except Exception as e:
            logger.warning("Failed to fetch Google Trends proxy: %s", e)
            return {"google_trends": 0.0}

    @staticmethod
    def _simple_sentiment(text: str) -> float:
        """Simple keyword-based sentiment scoring for social text.

        Returns: positive = bullish, negative = bearish, normalized to [-1, 1].
        """
        bullish_words = [
            "bull", "bullish", "moon", "pump", "breakout", "rally", "gains",
            "ath", "green", "buy", "buying", "long", "bounce", "surge",
            "bull run", "parabolic", "uptrend", "reversal up", "accumulation",
        ]
        bearish_words = [
            "bear", "bearish", "crash", "dump", "bleed", "red", "sell",
            "selling", "short", "drop", "plunge", "correction", "bubble",
            "fud", "panic", "capitulation", "downtrend", "dead", "scam",
            "hack", "exploit", "rug", "liquidation", "bankrupt",
        ]
        words = text.split()
        bull_count = sum(1 for w in words if w in bullish_words)
        bear_count = sum(1 for w in words if w in bearish_words)
        total = max(bull_count + bear_count, 1)
        return (bull_count - bear_count) / total


class OnChainIntelligence:
    """Collects on-chain and derivatives market data.

    Sources (all free):
    - Funding rates from various APIs
    - Open Interest changes
    - Exchange net flows
    - Liquidation data
    """

    def __init__(
        self,
        symbol: str = "BTC",
        coingecko_id: str = "bitcoin",
    ):
        self.symbol = symbol
        self.coingecko_id = coingecko_id

    async def collect_all(self) -> dict:
        """Gather all on-chain/derivatives data."""
        results = {}

        # Run independent fetches
        tasks = [
            self._fetch_funding_rates(),
            self._fetch_liquidations(),
            self._fetch_exchange_flows(),
        ]
        gathered = await asyncio.gather(*tasks, return_exceptions=True)
        for result in gathered:
            if isinstance(result, dict):
                results.update(result)
            elif isinstance(result, Exception):
                logger.warning("On-chain source error: %s", result)

        logger.info("On-chain intelligence collected: %s", results)
        return results

    async def _fetch_funding_rates(self) -> dict:
        """Funding rates from public APIs.

        Positive funding = longs paying shorts = bullish sentiment but over-leveraged.
        """
        try:
            async with aiohttp.ClientSession() as session:
                # CoinGlass free API alternative: use Binance public endpoint
                url = "https://fapi.binance.com/fapi/v1/premiumIndex"
                async with session.get(
                    url, params={"symbol": f"{self.symbol}USDT"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
                    funding_rate = float(data.get("lastFundingRate", 0))
                    mark_price = float(data.get("markPrice", 0))
                    logger.info("Funding rate %sUSDT: %.4f%%", self.symbol, funding_rate * 100)
                    return {
                        "funding_rate": funding_rate,
                        "mark_price": mark_price,
                    }
        except Exception as e:
            logger.warning("Failed to fetch funding rates: %s", e)
            return {"funding_rate": 0.0}

    async def _fetch_liquidations(self) -> dict:
        """Recent liquidation data from Binance futures."""
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://fapi.binance.com/fapi/v1/allForceOrders"
                params = {"symbol": f"{self.symbol}USDT", "limit": 50}
                async with session.get(
                    url, params=params, timeout=aiohttp.ClientTimeout(total=10)
                ) as resp:
                    orders = await resp.json()
                    buys = sum(
                        float(o.get("qty", 0))
                        for o in orders
                        if o.get("side") == "BUY"
                    )
                    sells = sum(
                        float(o.get("qty", 0))
                        for o in orders
                        if o.get("side") == "SELL"
                    )
                    total = max(buys + sells, 1)
                    # Positive = more short liquidations (good for price)
                    liq_score = (buys - sells) / total
                    logger.info(
                        "Liquidations buy_vol=%.0f sell_vol=%.0f score=%.3f",
                        buys, sells, liq_score,
                    )
                    return {
                        "liquidations_score": liq_score,
                        "liquidation_buy_vol": buys,
                        "liquidation_sell_vol": sells,
                    }
        except Exception as e:
            logger.warning("Failed to fetch liquidations: %s", e)
            return {"liquidations_score": 0.0}

    async def _fetch_exchange_flows(self) -> dict:
        """Exchange net flow proxy using on-chain metrics.

        Free source: DeFiLlama or Glassnode free tier.
        For now, use volume as a proxy indicator.
        """
        try:
            async with aiohttp.ClientSession() as session:
                url = "https://pro-api.coingecko.com/api/v3/coins/markets"
                # Try free endpoint first
                alt_url = "https://api.coingecko.com/api/v3/coins/markets"
                async with session.get(
                    alt_url,
                    params={"vs_currency": "usd", "ids": self.coingecko_id,
                            "order": "market_cap_desc"},
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    data = await resp.json()
                    if data:
                        coin = data[0]
                        vol_to_mcap = coin.get("total_volume", 0) / max(coin.get("market_cap", 1), 1)
                        # High vol/mcap ratio = increased activity, often precedes moves
                        flow_proxy = (vol_to_mcap - 0.05) / 0.1  # normalize around typical 0.05 baseline
                        flow_score = max(-1.0, min(1.0, flow_proxy))
                        return {
                            "exchange_flow": flow_score,
                            "vol_to_mcap_ratio": vol_to_mcap,
                        }
        except Exception as e:
            logger.warning("Failed to fetch exchange flows: %s", e)
            return {"exchange_flow": 0.0}


class MarketRegimeDetector:
    """Detects current market regime for risk assessment."""

    @staticmethod
    def detect(
        prices: list[float],
        volumes: list[float],
        correlation_btc: float = 0.0,
    ) -> dict:
        """Analyze market regime from price/volume data.

        Returns:
            dict with volatility_regime, volume_anomaly, btc_correlation
        """
        if len(prices) < 20:
            return {
                "volatility_regime": "normal",
                "volume_anomaly": False,
                "btc_correlation": correlation_btc,
            }

        import numpy as np

        # Volatility regime
        returns = np.diff(prices) / np.array(prices[:-1])
        vol = float(np.std(returns)) * np.sqrt(252 * 24)  # annualized for 1h

        if vol < 0.3:
            regime = "low"
        elif vol < 0.6:
            regime = "normal"
        elif vol < 1.0:
            regime = "high"
        else:
            regime = "extreme"

        # Volume anomaly detection
        vol_mean = np.mean(volumes[-20:])
        vol_std = max(np.std(volumes[-20:]), 1e-10)
        current_vol = volumes[-1] if volumes else vol_mean
        is_anomaly = abs(current_vol - vol_mean) > 2 * vol_std

        return {
            "volatility_regime": regime,
            "volume_anomaly": is_anomaly,
            "btc_correlation": correlation_btc,
            "annualized_vol": vol,
        }
