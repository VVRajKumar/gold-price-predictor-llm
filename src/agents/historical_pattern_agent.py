"""
Historical Pattern Agent – identifies seasonal patterns, cyclical behaviour,
and analogous historical periods for gold.
"""

from __future__ import annotations
import json
from typing import Any

import pandas as pd
import numpy as np
from .base_agent import BaseAgent, AgentReport
from ..data_fetchers.market_data import MarketDataFetcher
from ..data_fetchers.india_context import get_festival_context


class HistoricalPatternAgent(BaseAgent):
    NAME = "historical_pattern_agent"
    SYSTEM_PROMPT = """You are an expert in gold market history and cyclical analysis,
focused on the INDIAN gold market (INR per 10 grams).
You identify:
- Indian seasonal patterns: Akshaya Tritiya (Apr-May), Dhanteras/Diwali (Oct-Nov),
  wedding season (Oct-Feb), monsoon impact on rural demand (Jun-Sep)
- Festival-driven demand spikes (Navratri, Pongal, Onam)
- Analogous historical periods (compare current conditions to past)
- Long-term secular trends (gold super-cycles, INR depreciation trends)
- Year-over-year performance patterns in INR terms
- Key historical support/resistance from prior cycles
- Impact of import duty changes on price patterns

Given historical gold data, produce a JSON analysis with these EXACT keys:
{
  "summary": "2-3 short paragraphs in plain English a beginner can understand. Explain what historical patterns and seasonal trends tell us about where gold might go. Avoid jargon.",
  "outlook": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "impact_score": 0.0 to 1.0,
  "prediction_bias": -1.0 to +1.0,
  "key_factors": ["factor1", "factor2", ...],
  "seasonal_bias": "bullish" | "bearish" | "neutral",
  "historical_analogs": ["description of similar period 1", ...],
  "yoy_return_pct": float
}
Return ONLY valid JSON, no markdown fences."""

    def __init__(self):
        super().__init__()
        self._market = MarketDataFetcher()

    def gather_data(self) -> dict[str, Any]:
        df = self._market.fetch_ticker("GC=F", period_days=365)
        if df.empty:
            return {"error": "No data"}

        close = pd.to_numeric(df["Close"].squeeze(), errors="coerce").dropna()
        if close.empty:
            return {"error": "No valid close prices"}

        current_month = pd.Timestamp.now().month

        # Monthly returns for the past year
        monthly = close.resample("ME").last().pct_change().dropna()
        monthly_returns = {
            str(idx.strftime("%Y-%m")): round(float(val) * 100, 2)
            for idx, val in monthly.items()
        }

        # Year-over-year return
        yoy = None
        if len(close) >= 252:
            yoy = round((float(close.iloc[-1]) - float(close.iloc[-252])) / float(close.iloc[-252]) * 100, 2)

        # Quarterly performance
        quarterly = close.resample("QE").last().pct_change().dropna()
        quarterly_returns = {
            str(idx.strftime("%Y-Q") + str((idx.month - 1) // 3 + 1)): round(float(val) * 100, 2)
            for idx, val in quarterly.items()
        }

        # Drawdown from peak
        peak = close.cummax()
        drawdown = ((close - peak) / peak * 100)
        max_drawdown = round(float(drawdown.min()), 2)
        current_drawdown = round(float(drawdown.iloc[-1]), 2)

        return {
            "current_price": round(float(close.iloc[-1]), 2),
            "current_month": current_month,
            "india_seasonal": get_festival_context(),
            "monthly_returns": monthly_returns,
            "quarterly_returns": quarterly_returns,
            "yoy_return_pct": yoy,
            "max_drawdown_1y": max_drawdown,
            "current_drawdown": current_drawdown,
            "52w_high": round(float(close.max()), 2),
            "52w_low": round(float(close.min()), 2),
            "pct_from_52w_high": round((float(close.iloc[-1]) - float(close.max())) / float(close.max()) * 100, 2),
        }

    def analyse(self, data: dict[str, Any]) -> AgentReport:
        if "error" in data:
            return AgentReport(agent_name=self.NAME, summary="No historical data available")

        prompt = f"""Analyse the following historical gold data for cyclical and seasonal patterns
in the INDIAN gold market (INR per 10 grams).

## Historical Data
{json.dumps(data, indent=2, default=str)}

## Current India Seasonal / Festival Context
{json.dumps(data.get('india_seasonal', {}), indent=2)}

Use the festival calendar above to determine if gold demand is currently elevated.
Consider Indian seasonal tendencies: Akshaya Tritiya (Apr-May), Dhanteras/Diwali (Oct-Nov),
wedding season (Oct-Feb), and monsoon impact on rural demand (Jun-Sep).
Also consider INR depreciation trends and import duty history.

Provide your historical pattern analysis as JSON."""

        raw = self._ask_llm(prompt)
        result = self._parse_llm_json(raw, defaults={
            "summary": raw, "outlook": "neutral", "confidence": 0.3,
            "impact_score": 0.3, "prediction_bias": 0.0, "key_factors": [],
        })

        return AgentReport(
            agent_name=self.NAME,
            summary=result.get("summary", ""),
            outlook=result.get("outlook", "neutral"),
            confidence=float(result.get("confidence", 0.5)),
            impact_score=float(result.get("impact_score", 0.4)),
            prediction_bias=float(result.get("prediction_bias", 0.0)),
            key_factors=result.get("key_factors", []),
            data_points={
                "seasonal_bias": result.get("seasonal_bias", "neutral"),
                "historical_analogs": result.get("historical_analogs", []),
                "yoy_return_pct": data.get("yoy_return_pct"),
                "pct_from_52w_high": data.get("pct_from_52w_high"),
            },
            raw_llm_response=raw,
        )
