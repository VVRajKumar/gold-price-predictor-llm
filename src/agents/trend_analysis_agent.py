"""
Trend Analysis Agent – analyses gold price trends using recent price action.
"""

from __future__ import annotations
import json
from typing import Any

import pandas as pd
from .base_agent import BaseAgent, AgentReport
from ..data_fetchers.market_data import MarketDataFetcher


class TrendAnalysisAgent(BaseAgent):
    NAME = "trend_analysis_agent"
    SYSTEM_PROMPT = """You are a senior commodities trader specialising in gold price trends.
You analyse price data, moving averages, momentum, and rate-of-change to determine
the prevailing trend direction and strength.

Given recent gold price data, produce a JSON analysis with these EXACT keys:
{
  "summary": "2-3 paragraph trend analysis",
  "outlook": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "impact_score": 0.0 to 1.0,
  "prediction_bias": -1.0 to +1.0,
  "key_factors": ["factor1", "factor2", ...],
  "trend_strength": "strong" | "moderate" | "weak",
  "support_levels": [price1, price2],
  "resistance_levels": [price1, price2],
  "7d_direction": "up" | "down" | "sideways"
}
Return ONLY valid JSON, no markdown fences."""

    def __init__(self):
        super().__init__()
        self._market = MarketDataFetcher()

    def gather_data(self) -> dict[str, Any]:
        summary = self._market.get_gold_summary(period_days=90)
        df = self._market.fetch_ticker("GC=F", period_days=90)
        correlations = self._market.get_correlation_snapshot(90)

        # Compute simple moving averages
        sma_data = {}
        if not df.empty:
            close = df["Close"].squeeze()
            for window in [5, 10, 20, 50]:
                sma = close.rolling(window).mean()
                if len(sma.dropna()) > 0:
                    sma_data[f"SMA_{window}"] = round(float(sma.iloc[-1]), 2)

            # Recent price action
            sma_data["last_5_closes"] = [round(float(c), 2) for c in close.tail(5).tolist()]
            sma_data["last_20_closes"] = [round(float(c), 2) for c in close.tail(20).tolist()]

        return {
            "gold_summary": summary,
            "sma_data": sma_data,
            "correlations": correlations,
        }

    def analyse(self, data: dict[str, Any]) -> AgentReport:
        prompt = f"""Analyse the following gold price data and determine the current trend.

## Gold Summary
{json.dumps(data.get('gold_summary', {}), indent=2)}

## Moving Averages & Recent Prices
{json.dumps(data.get('sma_data', {}), indent=2)}

## Correlations with Other Assets
{json.dumps(data.get('correlations', {}), indent=2)}

Provide your trend analysis as JSON."""

        raw = self._ask_llm(prompt)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "summary": raw, "outlook": "neutral", "confidence": 0.4,
                "impact_score": 0.5, "prediction_bias": 0.0, "key_factors": [],
            }

        return AgentReport(
            agent_name=self.NAME,
            summary=result.get("summary", ""),
            outlook=result.get("outlook", "neutral"),
            confidence=float(result.get("confidence", 0.5)),
            impact_score=float(result.get("impact_score", 0.6)),
            prediction_bias=float(result.get("prediction_bias", 0.0)),
            key_factors=result.get("key_factors", []),
            data_points={
                "trend_strength": result.get("trend_strength", "moderate"),
                "support_levels": result.get("support_levels", []),
                "resistance_levels": result.get("resistance_levels", []),
                "7d_direction": result.get("7d_direction", "sideways"),
                **data.get("sma_data", {}),
            },
            raw_llm_response=raw,
        )
