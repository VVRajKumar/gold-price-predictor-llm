"""
Macro-Economics Agent – analyses RBI policy, inflation, INR, and global macro for Indian gold.
"""

from __future__ import annotations
import json
from typing import Any

from .base_agent import BaseAgent, AgentReport
from ..data_fetchers.macro_data import MacroDataFetcher
from ..data_fetchers.market_data import MarketDataFetcher
from ..data_fetchers.india_context import get_india_macro_context, get_festival_context


class MacroEconomicsAgent(BaseAgent):
    NAME = "macro_economics_agent"
    SYSTEM_PROMPT = """You are a senior macroeconomist specialising in how monetary policy
and economic indicators affect the INDIAN gold market (prices in INR per 10 grams).
Your expertise covers:
- RBI (Reserve Bank of India) policy: repo rate decisions, CRR, SLR, and forward guidance
- Indian inflation (CPI India) and its impact on gold demand
- USD/INR exchange rate – INR depreciation directly pushes Indian gold higher
- US Federal Reserve policy (global spillover to Indian markets)
- India's current account deficit – higher deficit weakens INR → bullish for gold
- Indian import duty on gold – duty hikes/cuts directly affect domestic prices
- Festival and wedding season demand (Dhanteras, Akshaya Tritiya, wedding season)
- RBI gold reserve purchases and sovereign gold bond issuance
- Global real interest rates – negative real rates are bullish for gold

Given the latest macro data, produce a JSON analysis with these EXACT keys:
{
  "summary": "2-3 paragraph macro analysis focused on Indian gold price implications",
  "outlook": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "impact_score": 0.0 to 1.0,
  "prediction_bias": -1.0 to +1.0,
  "key_factors": ["factor1", "factor2", ...],
  "rbi_rate_outlook": "cutting" | "holding" | "hiking",
  "inflation_trend": "rising" | "stable" | "falling",
  "inr_outlook": "strengthening" | "stable" | "weakening"
}
Return ONLY valid JSON, no markdown fences."""

    def __init__(self):
        super().__init__()
        self._macro = MacroDataFetcher()
        self._market = MarketDataFetcher()

    def gather_data(self) -> dict[str, Any]:
        macro_summary = self._macro.get_macro_summary()

        # USD/INR exchange rate
        usdinr_df = self._market.fetch_ticker("INR=X", period_days=90)
        usdinr_info = {}
        if not usdinr_df.empty:
            usdinr_close = usdinr_df["Close"].squeeze()
            usdinr_info = {
                "current_rate": round(float(usdinr_close.iloc[-1]), 2),
                "30d_change": round(
                    float(usdinr_close.iloc[-1]) - float(usdinr_close.iloc[-min(30, len(usdinr_close))]),
                    2,
                ),
                "90d_change": round(
                    float(usdinr_close.iloc[-1]) - float(usdinr_close.iloc[0]),
                    2,
                ),
            }

        # DXY (US Dollar Index) – affects INR indirectly
        dxy_df = self._market.fetch_ticker("DX-Y.NYB", period_days=90)
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

        # Nifty 50 for domestic macro sentiment
        nifty_df = self._market.fetch_ticker("^NSEI", period_days=30)
        nifty_info = {}
        if not nifty_df.empty:
            nifty_close = nifty_df["Close"].squeeze()
            nifty_info = {
                "current": round(float(nifty_close.iloc[-1]), 2),
                "30d_change_pct": round(
                    (float(nifty_close.iloc[-1]) - float(nifty_close.iloc[0]))
                    / float(nifty_close.iloc[0]) * 100, 2
                ),
            }

        # COMEX gold (global reference)
        comex_df = self._market.fetch_ticker("GC=F", period_days=30)
        comex_info = {}
        if not comex_df.empty:
            close = comex_df["Close"].squeeze()
            comex_info = {
                "symbol": "GC=F",
                "current_usd": round(float(close.iloc[-1]), 2),
                "7d_change": round(float(close.iloc[-1]) - float(close.iloc[-min(7, len(close))]), 2),
                "30d_change": round(float(close.iloc[-1]) - float(close.iloc[0]), 2),
            }

        return {
            "macro_summary": macro_summary,
            "india_macro": get_india_macro_context(),
            "india_seasonal": get_festival_context(),
            "usdinr": usdinr_info,
            "dxy_index": dxy_info,
            "nifty50": nifty_info,
            "comex_gold": comex_info,
        }

    def analyse(self, data: dict[str, Any]) -> AgentReport:
        prompt = f"""Analyse the following macroeconomic data and its implications for INDIAN gold prices (INR).

## India-Specific Macro (RBI, CPI, Import Duty)
{json.dumps(data.get('india_macro', {}), indent=2)}

## India Seasonal / Festival Context
{json.dumps(data.get('india_seasonal', {}), indent=2)}

## Global Macro Indicators (FRED – US reference, affects India via USD)
{json.dumps(data.get('macro_summary', {}), indent=2)}

## USD/INR Exchange Rate
{json.dumps(data.get('usdinr', {}), indent=2)}

## US Dollar Index (DXY) – affects INR
{json.dumps(data.get('dxy_index', {}), indent=2)}

## Nifty 50 (Indian Equity Market)
{json.dumps(data.get('nifty50', {}), indent=2)}

## COMEX Gold (Global Reference in USD)
{json.dumps(data.get('comex_gold', {}), indent=2)}

Focus on how these factors affect gold prices in India (INR per 10 grams).
Consider the dual impact: global gold price moves AND INR exchange rate changes.
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
                "rbi_rate_outlook": result.get("rbi_rate_outlook", "holding"),
                "inflation_trend": result.get("inflation_trend", "stable"),
                "inr_outlook": result.get("inr_outlook", "stable"),
                **data.get("usdinr", {}),
                **data.get("dxy_index", {}),
                **data.get("comex_gold", {}),
            },
            raw_llm_response=raw,
        )
