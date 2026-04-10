"""
Geopolitics Agent – analyses geopolitical events and their impact on gold.
"""

from __future__ import annotations
from typing import Any

from .base_agent import BaseAgent, AgentReport
from ..data_fetchers.news_data import NewsDataFetcher


class GeopoliticsAgent(BaseAgent):
    NAME = "geopolitics_agent"
    SYSTEM_PROMPT = """You are a global-macro strategist analysing how international
political and economic developments affect the Indian gold market (prices in INR).
You will be given recent news headlines as data for your financial analysis.

Areas of focus:
- Diplomatic relations, trade negotiations, ceasefire agreements, and economic-policy shifts
- Central-bank reserve management (especially RBI gold purchases)
- Multilateral groups (BRICS, G-20) and currency-diversification trends
- Regional stability and Indian economic confidence
- Monsoon-season dynamics and rural gold demand in India
- Indian import-duty changes and government gold policy

Respond with a JSON object containing these keys:
  "summary" (string): 2-3 paragraph analysis of the current global landscape affecting Indian gold prices. Be specific — reference actual events from the headlines provided.
  "outlook" (string): one of "bullish", "bearish", or "neutral".
  "confidence" (number): 0.0 to 1.0.
  "impact_score" (number): 0.0 to 1.0 – how much global events are influencing Indian gold right now.
  "prediction_bias" (number): -1.0 to +1.0 – negative means bearish, positive means bullish for Indian gold.
  "key_factors" (array of strings): the main drivers — cite specific events from headlines.
  "risk_events" (array of strings): upcoming events that could cause a gold price move.

Respond with valid JSON only."""

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
        # Build individual headline strings, truncating early to control length
        geo_articles = data.get("geopolitics_news", [])[:20]
        gold_articles = data.get("gold_safe_haven_news", [])[:8]

        raw_headlines = [
            f"- [{a.get('source', '')}] {a.get('title', '')[:150]}"
            for a in geo_articles if a.get("title")
        ]
        raw_gold_headlines = [
            f"- [{a.get('source', '')}] {a.get('title', '')[:150]}"
            for a in gold_articles if a.get("title")
        ]

        # Sanitize external headlines to reduce content-filter triggers
        headlines = self._sanitize_headlines(
            "\n".join(raw_headlines), max_chars_per_line=140
        )
        gold_headlines = self._sanitize_headlines(
            "\n".join(raw_gold_headlines), max_chars_per_line=140
        )

        # Use explicit data boundary markers so Azure's jailbreak classifier
        # recognises these as structured data, not instructions.
        prompt = (
            "Analyse the following recent news headlines (provided as data for "
            "financial analysis) and assess their impact on Indian gold prices.\n"
            "Be specific — reference actual events such as ceasefire agreements, "
            "policy changes, or diplomatic developments mentioned in the headlines.\n\n"
            "[DATA: Global developments]\n"
            f"{headlines or 'No recent global news available.'}\n"
            "[END DATA]\n\n"
            "[DATA: Gold safe-haven headlines]\n"
            f"{gold_headlines or 'No safe-haven news available.'}\n"
            "[END DATA]\n\n"
            "Provide your analysis as JSON."
        )

        raw = self._ask_llm(prompt)

        # If the LLM returned an error, use a clean fallback instead of
        # propagating the raw error text to the UI.
        is_error = not isinstance(raw, str) or raw.startswith("ERROR:")

        if is_error:
            result = {
                "summary": (
                    "Geopolitical analysis temporarily unavailable; "
                    "defaulting to neutral outlook."
                ),
                "outlook": "neutral",
                "confidence": 0.3,
                "impact_score": 0.3,
                "prediction_bias": 0.0,
                "key_factors": [],
                "risk_events": [],
            }
        else:
            result = self._parse_llm_json(raw, defaults={
                "summary": raw,
                "outlook": "neutral",
                "confidence": 0.3,
                "impact_score": 0.3,
                "prediction_bias": 0.0,
                "key_factors": [],
                "risk_events": [],
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
                "risk_events": result.get("risk_events", []),
                "articles_analysed": len(geo_articles),
                "geopolitics_headlines_used": [a.get("title", "") for a in geo_articles if a.get("title")],
                "safe_haven_headlines_used": [a.get("title", "") for a in gold_articles if a.get("title")],
            },
            raw_llm_response=raw,
        )
