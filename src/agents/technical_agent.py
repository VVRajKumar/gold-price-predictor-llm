"""
Technical Analysis Agent – RSI, MACD, Bollinger Bands, Fibonacci, etc.
"""

from __future__ import annotations
import json
from typing import Any

import pandas as pd
import numpy as np
from src.agents.base_agent import BaseAgent, AgentReport
from src.data_fetchers.market_data import MarketDataFetcher


def _compute_rsi(series: pd.Series, period: int = 14) -> float:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0.0)).rolling(period).mean()
    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return round(float(rsi.iloc[-1]), 2) if not rsi.empty else 50.0


def _compute_macd(series: pd.Series) -> dict:
    ema12 = series.ewm(span=12).mean()
    ema26 = series.ewm(span=26).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9).mean()
    histogram = macd_line - signal_line
    return {
        "macd": round(float(macd_line.iloc[-1]), 4),
        "signal": round(float(signal_line.iloc[-1]), 4),
        "histogram": round(float(histogram.iloc[-1]), 4),
        "crossover": "bullish" if float(histogram.iloc[-1]) > 0 and float(histogram.iloc[-2]) <= 0
        else "bearish" if float(histogram.iloc[-1]) < 0 and float(histogram.iloc[-2]) >= 0
        else "none",
    }


def _compute_bollinger(series: pd.Series, window: int = 20, num_std: float = 2.0) -> dict:
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + (std * num_std)
    lower = sma - (std * num_std)
    current = float(series.iloc[-1])
    return {
        "upper_band": round(float(upper.iloc[-1]), 2),
        "middle_band": round(float(sma.iloc[-1]), 2),
        "lower_band": round(float(lower.iloc[-1]), 2),
        "bandwidth_pct": round((float(upper.iloc[-1]) - float(lower.iloc[-1])) / float(sma.iloc[-1]) * 100, 2),
        "position": (
            "above_upper" if current > float(upper.iloc[-1])
            else "below_lower" if current < float(lower.iloc[-1])
            else "within_bands"
        ),
    }


class TechnicalAnalysisAgent(BaseAgent):
    NAME = "technical_analysis_agent"
    SYSTEM_PROMPT = """You are a senior technical analyst specialising in gold.
You interpret RSI, MACD, Bollinger Bands, moving averages, support/resistance,
and candlestick patterns to forecast short-term gold price movement.

Given technical indicator data, produce a JSON analysis with these EXACT keys:
{
  "summary": "2-3 paragraph technical analysis",
  "outlook": "bullish" | "bearish" | "neutral",
  "confidence": 0.0 to 1.0,
  "impact_score": 0.0 to 1.0,
  "prediction_bias": -1.0 to +1.0,
  "key_factors": ["factor1", "factor2", ...],
  "key_levels": {"support": [p1, p2], "resistance": [p1, p2]},
  "signals": {"rsi": "overbought|neutral|oversold", "macd": "bullish|bearish|neutral", "bollinger": "..."}
}
Return ONLY valid JSON, no markdown fences."""

    def __init__(self):
        super().__init__()
        self._market = MarketDataFetcher()

    def gather_data(self) -> dict[str, Any]:
        df = self._market.fetch_ticker("GC=F", period_days=120)
        if df.empty:
            return {"error": "No gold data"}

        close = df["Close"].squeeze()
        high = df["High"].squeeze()
        low = df["Low"].squeeze()
        indicators = {
            "rsi_14": _compute_rsi(close, 14),
            "macd": _compute_macd(close),
            "bollinger": _compute_bollinger(close),
            "sma_20": round(float(close.rolling(20).mean().iloc[-1]), 2),
            "sma_50": round(float(close.rolling(50).mean().iloc[-1]), 2),
            "ema_9": round(float(close.ewm(span=9).mean().iloc[-1]), 2),
            "current_price": round(float(close.iloc[-1]), 2),
            "last_10_closes": [round(float(c), 2) for c in close.tail(10).tolist()],
            "atr_14": round(float(
                pd.concat([
                    high - low,
                    (high - close.shift()).abs(),
                    (low - close.shift()).abs(),
                ], axis=1).max(axis=1).rolling(14).mean().iloc[-1]
            ), 2),
        }

        return {"indicators": indicators}

    def analyse(self, data: dict[str, Any]) -> AgentReport:
        if "error" in data:
            return AgentReport(agent_name=self.NAME, summary="No data available")

        prompt = f"""Analyse the following technical indicators for gold (GC=F).

## Technical Indicators
{json.dumps(data.get('indicators', {}), indent=2)}

Provide your technical analysis as JSON."""

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
                "key_levels": result.get("key_levels", {}),
                "signals": result.get("signals", {}),
                **data.get("indicators", {}),
            },
            raw_llm_response=raw,
        )
