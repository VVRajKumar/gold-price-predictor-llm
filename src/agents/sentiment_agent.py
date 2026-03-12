"""
Sentiment Agent – analyses market sentiment across news, VIX, and fear/greed signals.
"""

from __future__ import annotations
import json
from typing import Any

from src.agents.base_agent import BaseAgent, AgentReport
from src.data_fetchers.market_data import MarketDataFetcher
from src.data_fetchers.news_data import NewsDataFetcher


class SentimentAgent(BaseAgent):
    NAME = "sentiment_agent"
    SYSTEM_PROMPT = """You are a senior market sentiment analyst. You gauge fear/greed
across markets and determine how sentiment is likely to drive gold prices.
You analyse:
- VIX (fear index) – high VIX = risk-off = gold-positive
- S&P 500 movements – sharp drops = flight to gold
- News sentiment (positive/negative tone of gold coverage)
- Market-wide risk appetite vs. safe-haven demand

Given sentiment data, produce a JSON analysis with these EXACT keys:
{
  "summary": "2-3 paragraph sentiment analysis for gold",
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
        # VIX data
        vix_df = self._market.fetch_ticker("^VIX", period_days=30)
        vix_info = {}
        if not vix_df.empty:
            vix_close = vix_df["Close"].squeeze()
            vix_info = {
                "current": round(float(vix_close.iloc[-1]), 2),
                "5d_avg": round(float(vix_close.tail(5).mean()), 2),
                "30d_avg": round(float(vix_close.mean()), 2),
                "is_elevated": float(vix_close.iloc[-1]) > 20,
            }

        # S&P 500 recent action
        sp_df = self._market.fetch_ticker("^GSPC", period_days=30)
        sp_info = {}
        if not sp_df.empty:
            sp_close = sp_df["Close"].squeeze()
            sp_info = {
                "5d_change_pct": round(
                    (float(sp_close.iloc[-1]) - float(sp_close.iloc[-min(5, len(sp_close))]))
                    / float(sp_close.iloc[-min(5, len(sp_close))]) * 100, 2
                ),
            }

        # News headlines for sentiment
        gold_news = self._news.fetch_newsapi(days_back=2, page_size=15)
        headlines = [a["title"] for a in gold_news[:15]]

        return {"vix_info": vix_info, "sp500_info": sp_info, "news_headlines": headlines}

    def analyse(self, data: dict[str, Any]) -> AgentReport:
        prompt = f"""Analyse market sentiment and its implications for gold.

## VIX (Fear Index)
{json.dumps(data.get('vix_info', {}), indent=2)}

## S&P 500 Recent Performance
{json.dumps(data.get('sp500_info', {}), indent=2)}

## Recent Gold-Related Headlines (gauge sentiment)
{chr(10).join(['- ' + h for h in data.get('news_headlines', [])])}

Provide your sentiment analysis as JSON."""

        raw = self._ask_llm(prompt)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "summary": raw, "outlook": "neutral", "confidence": 0.4,
                "impact_score": 0.4, "prediction_bias": 0.0, "key_factors": [],
            }

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
                **data.get("vix_info", {}),
            },
            raw_llm_response=raw,
        )
