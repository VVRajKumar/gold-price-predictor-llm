"""
Geopolitics Agent – analyses geopolitical events and their impact on gold.
"""

from __future__ import annotations
import json
from typing import Any

from src.agents.base_agent import BaseAgent, AgentReport
from src.data_fetchers.news_data import NewsDataFetcher


class GeopoliticsAgent(BaseAgent):
    NAME = "geopolitics_agent"
    SYSTEM_PROMPT = """You are a senior geopolitical analyst specialising in how global
events affect the gold market. You have deep expertise in:
- Military conflicts, wars, and territorial disputes
- Economic sanctions and trade wars
- Central bank gold reserve changes
- BRICS alliance and de-dollarisation efforts
- Political instability and regime changes
- Refugee crises and humanitarian emergencies

Given the latest geopolitical news, produce a JSON analysis with these EXACT keys:
{
  "summary": "2-3 paragraph analysis of the current geopolitical landscape affecting gold",
  "outlook": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "impact_score": 0.0 to 1.0 (how much geopolitics is moving gold RIGHT NOW),
  "prediction_bias": -1.0 to +1.0 (-1 = very bearish, +1 = very bullish for gold),
  "key_factors": ["factor1", "factor2", ...],
  "risk_events": ["upcoming event that could cause gold spike/drop", ...]
}
Return ONLY valid JSON, no markdown fences."""

    def __init__(self):
        super().__init__()
        self._news = NewsDataFetcher()

    def gather_data(self) -> dict[str, Any]:
        news = self._news.fetch_geopolitics_news(days_back=3)
        gold_news = self._news.fetch_newsapi(
            query="gold safe haven geopolitical risk", days_back=3, page_size=10
        )
        return {"geopolitics_news": news, "gold_safe_haven_news": gold_news}

    def analyse(self, data: dict[str, Any]) -> AgentReport:
        headlines = "\n".join(
            [f"- [{a['source']}] {a['title']}" for a in data.get("geopolitics_news", [])[:25]]
        )
        gold_headlines = "\n".join(
            [f"- [{a['source']}] {a['title']}" for a in data.get("gold_safe_haven_news", [])[:10]]
        )

        prompt = f"""Analyse the following recent geopolitical news and their impact on gold prices.

## Geopolitical Headlines
{headlines or "No recent geopolitics news available."}

## Gold Safe-Haven Headlines
{gold_headlines or "No safe-haven news available."}

Provide your analysis as JSON."""

        raw = self._ask_llm(prompt)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "summary": raw,
                "outlook": "neutral",
                "confidence": 0.3,
                "impact_score": 0.3,
                "prediction_bias": 0.0,
                "key_factors": [],
                "risk_events": [],
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
                "risk_events": result.get("risk_events", []),
                "articles_analysed": len(data.get("geopolitics_news", [])),
            },
            raw_llm_response=raw,
        )
