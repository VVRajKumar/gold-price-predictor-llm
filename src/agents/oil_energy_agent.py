"""
Oil & Energy Agent – analyses oil prices and energy markets' impact on gold.
"""

from __future__ import annotations
import json
from typing import Any

from .base_agent import BaseAgent, AgentReport
from ..data_fetchers.market_data import MarketDataFetcher
from ..data_fetchers.news_data import NewsDataFetcher


class OilEnergyAgent(BaseAgent):
    NAME = "oil_energy_agent"
    SYSTEM_PROMPT = """You are a senior energy market analyst specialising in the
relationship between oil prices and gold. You understand:
- Oil-gold ratio and historical correlations
- OPEC decisions and supply disruptions
- Energy-driven inflation pass-through to gold
- Petrodollar dynamics and their gold implications
- Energy crises as a safe-haven catalyst for gold

Given oil & energy data, produce a JSON analysis with these EXACT keys:
{
  "summary": "2-3 paragraph analysis of energy markets' gold impact",
  "outlook": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "impact_score": 0.0 to 1.0,
  "prediction_bias": -1.0 to +1.0,
  "key_factors": ["factor1", "factor2", ...],
  "oil_gold_ratio": float,
  "energy_inflation_risk": "high" | "moderate" | "low"
}
Return ONLY valid JSON, no markdown fences."""

    def __init__(self):
        super().__init__()
        self._market = MarketDataFetcher()
        self._news = NewsDataFetcher()

    def gather_data(self) -> dict[str, Any]:
        oil_df = self._market.fetch_ticker("CL=F", period_days=90)
        gold_df = self._market.fetch_ticker("GC=F", period_days=90)

        oil_info = {}
        if not oil_df.empty:
            oil_close = oil_df["Close"].squeeze()
            oil_high = oil_df["High"].squeeze()
            oil_low = oil_df["Low"].squeeze()
            oil_info = {
                "current_price": round(float(oil_close.iloc[-1]), 2),
                "30d_high": round(float(oil_high.tail(30).max()), 2),
                "30d_low": round(float(oil_low.tail(30).min()), 2),
                "30d_change_pct": round(
                    (float(oil_close.iloc[-1]) - float(oil_close.iloc[-min(30, len(oil_close))]))
                    / float(oil_close.iloc[-min(30, len(oil_close))]) * 100, 2
                ),
            }

        gold_oil_ratio = None
        if not oil_df.empty and not gold_df.empty:
            gold_oil_ratio = round(
                float(gold_df["Close"].squeeze().iloc[-1]) / float(oil_df["Close"].squeeze().iloc[-1]), 2
            )

        energy_news = self._news.fetch_newsapi(
            query="oil OPEC energy prices crude", days_back=3, page_size=10
        )

        return {
            "oil_info": oil_info,
            "gold_oil_ratio": gold_oil_ratio,
            "energy_news_headlines": [a["title"] for a in energy_news[:10]],
        }

    def analyse(self, data: dict[str, Any]) -> AgentReport:
        prompt = f"""Analyse the following oil/energy data and its impact on gold.

## Oil Market Data
{json.dumps(data.get('oil_info', {}), indent=2)}

## Gold-Oil Ratio
{data.get('gold_oil_ratio', 'N/A')}  (Historical average ~15-25)

## Energy News Headlines
{chr(10).join(['- ' + h for h in data.get('energy_news_headlines', [])])}

Provide your analysis as JSON."""

        raw = self._ask_llm(prompt)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "summary": raw, "outlook": "neutral", "confidence": 0.4,
                "impact_score": 0.3, "prediction_bias": 0.0, "key_factors": [],
            }

        return AgentReport(
            agent_name=self.NAME,
            summary=result.get("summary", ""),
            outlook=result.get("outlook", "neutral"),
            confidence=float(result.get("confidence", 0.5)),
            impact_score=float(result.get("impact_score", 0.4)),
            prediction_bias=float(result.get("prediction_bias", 0.0)),
            key_factors=result.get("key_factors", []),
            data_points={
                "oil_gold_ratio": data.get("gold_oil_ratio"),
                "energy_inflation_risk": result.get("energy_inflation_risk", "moderate"),
                **data.get("oil_info", {}),
            },
            raw_llm_response=raw,
        )
