"""
Macro-Economics Agent – analyses interest rates, inflation, USD, money supply.
"""

from __future__ import annotations
import json
from typing import Any

from .base_agent import BaseAgent, AgentReport
from ..data_fetchers.macro_data import MacroDataFetcher
from ..data_fetchers.market_data import MarketDataFetcher


class MacroEconomicsAgent(BaseAgent):
    NAME = "macro_economics_agent"
    SYSTEM_PROMPT = """You are a senior macroeconomist specialising in how monetary policy
and economic indicators affect the gold market. Your expertise covers:
- Federal Reserve policy, interest rate decisions, and forward guidance
- Real interest rates (the single most important macro driver of gold)
- Inflation expectations (CPI, PCE, breakeven rates)
- US Dollar strength (DXY) – inversely correlated with gold
- Money supply (M2) – expansion is bullish for gold
- US national debt – higher debt is bullish for gold long-term
- Recession probability indicators

Given the latest macro data, produce a JSON analysis with these EXACT keys:
{
  "summary": "2-3 paragraph macro analysis focused on gold implications",
  "outlook": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "impact_score": 0.0 to 1.0,
  "prediction_bias": -1.0 to +1.0,
  "key_factors": ["factor1", "factor2", ...],
  "rate_outlook": "cutting" | "holding" | "hiking",
  "inflation_trend": "rising" | "stable" | "falling",
  "usd_outlook": "strengthening" | "stable" | "weakening"
}
Return ONLY valid JSON, no markdown fences."""

    def __init__(self):
        super().__init__()
        self._macro = MacroDataFetcher()
        self._market = MarketDataFetcher()

    def gather_data(self) -> dict[str, Any]:
        macro_summary = self._macro.get_macro_summary()
        # Also get DXY and Treasury yield snapshot
        dxy_df = self._market.fetch_ticker("DX-Y.NYB", period_days=90)
        tnx_df = self._market.fetch_ticker("^TNX", period_days=90)

        dxy_info = {}
        if not dxy_df.empty:
            dxy_close = dxy_df["Close"].squeeze()
            dxy_info = {
                "current": round(float(dxy_close.iloc[-1]), 2),
                "30d_change": round(
                    float(dxy_close.iloc[-1]) - float(dxy_close.iloc[-min(30, len(dxy_close))]),
                    2,
                ),
            }

        tnx_info = {}
        if not tnx_df.empty:
            tnx_close = tnx_df["Close"].squeeze()
            tnx_info = {
                "current_yield": round(float(tnx_close.iloc[-1]), 3),
                "30d_change": round(
                    float(tnx_close.iloc[-1]) - float(tnx_close.iloc[-min(30, len(tnx_close))]),
                    3,
                ),
            }

        return {
            "macro_summary": macro_summary,
            "usd_index": dxy_info,
            "treasury_10y": tnx_info,
        }

    def analyse(self, data: dict[str, Any]) -> AgentReport:
        prompt = f"""Analyse the following macroeconomic data and its implications for gold.

## FRED Macro Indicators
{json.dumps(data.get('macro_summary', {}), indent=2)}

## US Dollar Index (DXY)
{json.dumps(data.get('usd_index', {}), indent=2)}

## 10-Year Treasury Yield
{json.dumps(data.get('treasury_10y', {}), indent=2)}

Provide your macro analysis as JSON."""

        raw = self._ask_llm(prompt)
        try:
            result = json.loads(raw)
        except json.JSONDecodeError:
            result = {
                "summary": raw, "outlook": "neutral", "confidence": 0.4,
                "impact_score": 0.6, "prediction_bias": 0.0, "key_factors": [],
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
                "rate_outlook": result.get("rate_outlook", "holding"),
                "inflation_trend": result.get("inflation_trend", "stable"),
                "usd_outlook": result.get("usd_outlook", "stable"),
                **data.get("usd_index", {}),
                **data.get("treasury_10y", {}),
            },
            raw_llm_response=raw,
        )
