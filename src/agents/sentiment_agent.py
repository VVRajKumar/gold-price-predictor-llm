"""
Sentiment Agent – analyses market sentiment across news, VIX, and fear/greed signals.
"""

from __future__ import annotations
import json
from typing import Any

from .base_agent import BaseAgent, AgentReport
from ..data_fetchers.market_data import MarketDataFetcher
from ..data_fetchers.news_data import NewsDataFetcher


class SentimentAgent(BaseAgent):
    NAME = "sentiment_agent"
    SYSTEM_PROMPT = """You are a senior market sentiment analyst focused on the INDIAN gold market.
You gauge fear/greed across Indian and global markets and determine how sentiment
is likely to drive gold prices in India (INR). You analyse:
- India VIX (fear index) – high India VIX = risk-off = gold-positive
- Nifty 50 / Sensex movements – sharp drops = flight to gold
- USD/INR exchange rate – INR weakening = higher Indian gold prices
- News sentiment (positive/negative tone of gold coverage in India)
- Festival and wedding season demand sentiment
- Market-wide risk appetite vs. safe-haven demand in India

Given sentiment data, produce a JSON analysis with these EXACT keys:
{
  "summary": "2-3 short paragraphs in plain English a beginner can understand. Explain what market mood (fear vs. greed) means for gold prices. Avoid jargon — briefly explain any financial terms in parentheses.",
  "outlook": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "impact_score": 0.0 to 1.0,
  "prediction_bias": -1.0 to +1.0,
  "key_factors": ["factor1", "factor2", ...],
  "fear_greed_level": "extreme_fear" | "fear" | "neutral" | "greed" | "extreme_greed",
  "safe_haven_demand": "very_high" | "high" | "moderate" | "low" | "very_low",
  "news_sentiment_score": -1.0 to 1.0
}
Return ONLY valid JSON, no markdown fences."""

    def __init__(self):
        super().__init__()
        self._market = MarketDataFetcher()
        self._news = NewsDataFetcher()

    def gather_data(self) -> dict[str, Any]:
        # India VIX data
        vix_df = self._market.fetch_ticker("^INDIAVIX", period_days=30)
        vix_info = {}
        if not vix_df.empty:
            vix_close = vix_df["Close"].squeeze()
            vix_info = {
                "index": "India VIX",
                "current": round(float(vix_close.iloc[-1]), 2),
                "5d_avg": round(float(vix_close.tail(5).mean()), 2),
                "30d_avg": round(float(vix_close.mean()), 2),
                "is_elevated": float(vix_close.iloc[-1]) > 18,
            }

        # Nifty 50 recent action
        nifty_df = self._market.fetch_ticker("^NSEI", period_days=30)
        nifty_info = {}
        if not nifty_df.empty:
            nifty_close = nifty_df["Close"].squeeze()
            _nifty_back = min(5, len(nifty_close) - 1) if len(nifty_close) > 1 else 0
            if _nifty_back > 0 and float(nifty_close.iloc[-1 - _nifty_back]) != 0:
                nifty_info = {
                    "index": "Nifty 50",
                    "5d_change_pct": round(
                        (float(nifty_close.iloc[-1]) - float(nifty_close.iloc[-1 - _nifty_back]))
                        / float(nifty_close.iloc[-1 - _nifty_back]) * 100, 2
                    ),
                }
            else:
                nifty_info = {"index": "Nifty 50", "5d_change_pct": 0.0}

        # USD/INR exchange rate
        usdinr_df = self._market.fetch_ticker("INR=X", period_days=30)
        usdinr_info = {}
        if not usdinr_df.empty:
            usdinr_close = usdinr_df["Close"].squeeze()
            _usd_back = min(5, len(usdinr_close) - 1) if len(usdinr_close) > 1 else 0
            usdinr_info = {
                "current_rate": round(float(usdinr_close.iloc[-1]), 2),
                "5d_change": round(
                    float(usdinr_close.iloc[-1]) - float(usdinr_close.iloc[-1 - _usd_back]) if _usd_back > 0 else 0.0, 2
                ),
            }

        # News headlines for sentiment
        gold_news = self._news.fetch_newsapi(days_back=2, page_size=15)
        headlines = [a["title"] for a in gold_news[:15]]

        return {
            "vix_info": vix_info,
            "nifty_info": nifty_info,
            "usdinr_info": usdinr_info,
            "news_headlines": headlines,
        }

    def analyse(self, data: dict[str, Any]) -> AgentReport:
        prompt = f"""Analyse market sentiment and its implications for INDIAN gold prices (INR).

## India VIX (Fear Index)
{json.dumps(data.get('vix_info', {}), indent=2)}

## Nifty 50 Recent Performance
{json.dumps(data.get('nifty_info', {}), indent=2)}

## USD/INR Exchange Rate
{json.dumps(data.get('usdinr_info', {}), indent=2)}

## Recent Gold-Related Headlines (gauge sentiment)
{chr(10).join(['- ' + h for h in data.get('news_headlines', [])])}

Provide your sentiment analysis focused on Indian gold market as JSON."""

        raw = self._ask_llm(prompt)
        result = self._parse_llm_json(raw, defaults={
            "summary": raw, "outlook": "neutral", "confidence": 0.4,
            "impact_score": 0.4, "prediction_bias": 0.0, "key_factors": [],
        })

        return AgentReport(
            agent_name=self.NAME,
            summary=result.get("summary", ""),
            outlook=result.get("outlook", "neutral"),
            confidence=float(result.get("confidence", 0.5)),
            impact_score=float(result.get("impact_score", 0.5)),
            prediction_bias=float(result.get("prediction_bias", 0.0)),
            key_factors=result.get("key_factors", []),
            data_points={
                "fear_greed_level": result.get("fear_greed_level", "neutral"),
                "safe_haven_demand": result.get("safe_haven_demand", "moderate"),
                "news_sentiment_score": result.get("news_sentiment_score", 0.0),
                "headlines_used": data.get("news_headlines", [])[:15],
                "headlines_count": len(data.get("news_headlines", []) or []),
                **data.get("vix_info", {}),
                **data.get("usdinr_info", {}),
            },
            raw_llm_response=raw,
        )
