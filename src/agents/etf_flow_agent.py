"""
ETF Flow Agent – analyses gold ETF inflows/outflows and mining stocks.
"""

from __future__ import annotations
import json
from typing import Any

from src.agents.base_agent import BaseAgent, AgentReport
from src.data_fetchers.etf_data import ETFDataFetcher


class ETFFlowAgent(BaseAgent):
    NAME = "etf_flow_agent"
    SYSTEM_PROMPT = """You are a senior ETF and fund-flow analyst specialising in gold.
You analyse gold ETF trading volume, price trends, and implied fund flows to gauge
institutional demand for gold.  Higher ETF inflows are bullish for gold;
outflows are bearish.

Given recent gold ETF data, produce a JSON analysis with these EXACT keys:
{
  "summary": "2-3 paragraph ETF flow analysis",
  "outlook": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "impact_score": 0.0 to 1.0,
  "prediction_bias": -1.0 to +1.0,
  "key_factors": ["factor1", "factor2", ...],
  "institutional_demand": "strong_buying" | "buying" | "neutral" | "selling" | "strong_selling",
  "top_movers": [{"ticker": "...", "signal": "..."}]
}
Return ONLY valid JSON, no markdown fences."""

    def __init__(self):
        super().__init__()
        self._etf = ETFDataFetcher()

    def gather_data(self) -> dict[str, Any]:
        etf_summary = self._etf.get_etf_flow_summary(period_days=30)
        return {"etf_summary": etf_summary}

    def analyse(self, data: dict[str, Any]) -> AgentReport:
        prompt = f"""Analyse the following gold ETF data for institutional fund-flow signals.

## Gold ETF Summary (30-day)
{json.dumps(data.get('etf_summary', {}), indent=2)}

Provide your ETF flow analysis as JSON."""

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
                "institutional_demand": result.get("institutional_demand", "neutral"),
                "top_movers": result.get("top_movers", []),
            },
            raw_llm_response=raw,
        )
