"""
LLM Narrator – generates human-readable narrative from ML predictions.

The narrator receives:
  * ML ensemble predictions (prices, bands, component breakdowns)
  * Agent reports (same 8 agents, run for intelligence gathering)
  * SHAP feature importances
  * Accuracy feedback

It produces:
  * executive_summary  – 3-4 paragraph synthesis
  * overall_outlook    – bullish / bearish / neutral
  * overall_confidence – 0-1 (derived from ML band width, NOT hallucinated)
  * bull_case / bear_case / risk_factors
  * per-hour key_driver strings

CRITICAL: The narrator NEVER generates price numbers.  All prices come from
the ML ensemble.  The LLM only writes prose + classifies outlook.
"""

from __future__ import annotations

import json
from typing import Any, Optional

import re

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger

from .config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
    TEMPERATURE, PREDICTION_HOURS, CACHE_DIR,
)
from .time_utils import now_ist

# ── Technical → human-friendly name replacements ─────────────────────
# Order matters: longer patterns first to avoid partial matches
_TECHNICAL_REPLACEMENTS = [
    ("roll_24", "24-hour moving average"),
    ("roll_12", "12-hour moving average"),
    ("roll_6", "short-term moving average"),
    ("lag_24", "24-hour price trend"),
    ("lag_12", "12-hour price trend"),
    ("lag_6", "6-hour price trend"),
    ("lag_3", "3-hour price trend"),
    ("lag_2", "short-term price trend"),
    ("lag_1", "recent price momentum"),
    ("ret_6h", "6-hour returns"),
    ("ret_1h", "1-hour returns"),
    ("vol_12", "recent volatility"),
    ("hour_sin", "market session timing"),
    ("hour_cos", "market session timing"),
    ("dow_sin", "day-of-week seasonality"),
    ("dow_cos", "day-of-week seasonality"),
]


def _sanitize_technical_names(result: dict) -> dict:
    """Replace any leftover internal feature codes with friendly names."""
    def _clean(text: str) -> str:
        for code, friendly in _TECHNICAL_REPLACEMENTS:
            text = text.replace(code, friendly)
        return text

    for key in ("executive_summary", "bull_case", "bear_case"):
        if key in result and isinstance(result[key], str):
            result[key] = _clean(result[key])

    if "risk_factors" in result and isinstance(result["risk_factors"], list):
        result["risk_factors"] = [_clean(r) if isinstance(r, str) else r for r in result["risk_factors"]]

    if "hourly_drivers" in result and isinstance(result["hourly_drivers"], list):
        result["hourly_drivers"] = [_clean(d) if isinstance(d, str) else d for d in result["hourly_drivers"]]

    return result


_NARRATOR_SYSTEM = """You are a friendly Gold Market Analyst writing a briefing that is easy to understand for everyday investors and beginners — not just finance professionals.

You are given:
1. ML model predictions (prices, confidence bands, component model outputs)
2. SHAP feature importance (which factors drove the ML prediction)
3. Eight specialist agent reports covering geopolitics, trend, ETF flows, macro,
   oil/energy, sentiment, technicals, and historical patterns
4. Recent accuracy feedback

YOUR ROLE: Narrate and explain the ML prediction.  You do NOT predict prices.
The ML models have already generated all the numbers.  Your job is to:
- Explain WHY the ML model predicts what it does (using SHAP + agent context)
- Provide an overall outlook classification (bullish / bearish / neutral)
- Write a bull case, bear case, and list risk factors
- For each prediction hour, write a brief key_driver string explaining
  what event/factor influences that hour (market sessions, data releases, etc.)

WRITING STYLE RULES:
- Write in plain, conversational English a 16-year-old could understand.
- Avoid jargon. If you must use a financial term, briefly explain it in parentheses.
- Use short sentences and short paragraphs (2-3 sentences each).
- Structure the executive_summary in 3 clear sections separated by newlines:
    1. "🔍 What's Happening:" — A 2-3 sentence overview of current gold price action.
    2. "📊 Why It Matters:" — 2-3 sentences explaining the main drivers in simple terms.
    3. "👀 What to Watch:" — 2-3 sentences on upcoming events or signals to monitor.
- For bull_case and bear_case, write short conversational paragraphs (3-4 sentences).
- For risk_factors, write each risk as a single clear sentence a beginner can understand.

CRITICAL NAMING RULES – NEVER use internal feature codes in your output.
Always use these human-readable names instead:
  lag_1 → "recent price momentum"      lag_2 → "short-term price trend"
  lag_3 → "3-hour price trend"         lag_6 → "6-hour price trend"
  lag_12 → "12-hour price trend"       lag_24 → "24-hour price trend"
  roll_6 → "short-term moving average" roll_12 → "12-hour moving average"
  roll_24 → "24-hour moving average"
  ret_1h → "1-hour returns"            ret_6h → "6-hour returns"
  vol_12 → "recent volatility"
  hour_sin / hour_cos → "time-of-day effects" or "market session timing"
  dow_sin / dow_cos → "day-of-week seasonality"
  sentiment_score → "market sentiment"  geopolitical_risk → "geopolitical risk"
  macro_outlook → "macro-economic outlook"  technical_signal → "technical indicators"
  etf_flow_signal → "ETF fund flows"   oil_energy_signal → "oil & energy impact"
  historical_seasonal → "seasonal patterns"  trend_strength → "trend strength"

Return ONLY valid JSON with these EXACT keys:
{
  "overall_outlook": "bullish" | "bearish" | "neutral",
  "executive_summary": "3-section summary using the format above",
  "bull_case": "conversational paragraph on the bullish scenario",
  "bear_case": "conversational paragraph on the bearish scenario",
  "risk_factors": ["risk1", "risk2", ...],
  "hourly_drivers": ["driver for hour 1", "driver for hour 2", ..., "driver for hour 24"]
}

Do NOT include any price numbers in your output (prices are from the ML model).
Do NOT wrap in markdown fences.
"""


_WEEKEND_NARRATOR_SYSTEM = """You are a friendly Gold Market Analyst writing a WEEKEND briefing that is easy to understand for everyday investors and beginners.

IMPORTANT CONTEXT: The gold market (MCX) is currently CLOSED for the weekend.
Trading will resume on Monday morning (09:00 IST).

You are given fresh intelligence from 8 specialist agents who have gathered the latest
news, geopolitical developments, economic data, and market signals DURING the weekend.
There are NO ML predictions because the market is closed — no trading is happening.

YOUR ROLE: Analyze what has happened over the weekend and predict what is likely to
happen when the market reopens on Monday.  Focus on:
- What news/events occurred during the weekend that could impact gold prices
- How these factors might affect the Monday opening price
- Whether gold is likely to gap up, gap down, or open near Friday's close

WRITING STYLE RULES:
- Write in plain, conversational English a 16-year-old could understand.
- Avoid jargon. If you must use a financial term, briefly explain it in parentheses.
- Use short sentences and short paragraphs (2-3 sentences each).
- Structure the executive_summary in 3 clear sections separated by newlines:
    1. "🔍 What's Happening:" — "The gold market is closed for the weekend." Then 2-3 sentences about weekend news/developments.
    2. "📊 Why It Matters:" — 2-3 sentences on how weekend events could affect Monday's opening.
    3. "👀 What to Watch on Monday:" — 2-3 sentences on what investors should monitor when the market reopens.
- For bull_case: what could push gold UP when the market reopens Monday.
- For bear_case: what could push gold DOWN when the market reopens Monday.
- For risk_factors: weekend events or Monday-morning catalysts that could cause surprises.

Return ONLY valid JSON with these EXACT keys:
{
  "overall_outlook": "bullish" | "bearish" | "neutral",
  "executive_summary": "3-section weekend summary using the format above",
  "bull_case": "what could push gold UP on Monday's opening",
  "bear_case": "what could push gold DOWN on Monday's opening",
  "risk_factors": ["risk1", "risk2", ...]
}

Do NOT include any price numbers in your output.
Do NOT wrap in markdown fences.
"""


class LLMNarrator:
    """Generates prose narrative from ML predictions + agent intelligence."""

    def __init__(self):
        self._llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=TEMPERATURE,
            request_timeout=90,
        )

    def narrate(
        self,
        ml_predictions: list[dict],
        agent_reports: dict[str, dict],
        shap_explanation: Optional[dict],
        current_price: float,
        feedback_context: str = "",
    ) -> dict[str, Any]:
        """
        Produce narrative content.

        Returns dict with keys: overall_outlook, executive_summary,
        bull_case, bear_case, risk_factors, hourly_drivers.
        """
        prompt = self._build_prompt(
            ml_predictions, agent_reports, shap_explanation,
            current_price, feedback_context,
        )

        try:
            response = self._llm.invoke([
                SystemMessage(content=_NARRATOR_SYSTEM),
                HumanMessage(content=prompt),
            ])
            result = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning("Narrator returned invalid JSON – using defaults")
            result = self._defaults()
        except Exception as e:
            logger.error(f"Narrator LLM call failed: {e}")
            result = self._defaults()

        # Validate
        outlook = str(result.get("overall_outlook", "neutral")).lower()
        if outlook not in {"bullish", "bearish", "neutral"}:
            outlook = "neutral"
        result["overall_outlook"] = outlook

        # Sanitize: replace any leftover technical feature names in all text fields
        result = _sanitize_technical_names(result)

        return result

    def narrate_weekend(
        self,
        agent_reports: dict[str, dict],
        current_price: float,
    ) -> dict[str, Any]:
        """Produce a weekend-specific narrative focused on Monday reopening.

        No ML predictions are used — agents provide fresh intelligence about
        weekend news/events and the LLM synthesizes a Monday outlook.
        """
        sections = [
            f"WEEKEND BRIEFING — Gold market is CLOSED.",
            f"Last Friday closing price: ₹{current_price:,.2f} per 10g.",
            f"Current day: {now_ist().strftime('%A %B %d, %Y')}.",
            "",
            "The following agent reports contain FRESH intelligence gathered during "
            "the weekend. Use these to analyze what will happen when the market "
            "reopens on Monday morning.",
        ]

        if agent_reports:
            sections.append("\n## Agent Intelligence Reports (Weekend Update)\n")
            for name, report in agent_reports.items():
                if not isinstance(report, dict):
                    continue
                bias = report.get('prediction_bias', 0)
                bias = bias if isinstance(bias, (int, float)) else 0
                conf = report.get('confidence', 0)
                conf = conf if isinstance(conf, (int, float)) else 0
                kf = report.get('key_factors', []) or []
                sections.append(
                    f"### {name.replace('_', ' ').title()}\n"
                    f"- Outlook: {report.get('outlook', 'N/A')} | "
                    f"Confidence: {conf:.2f} | "
                    f"Bias: {bias:+.2f}\n"
                    f"- Summary: {str(report.get('summary', ''))[:500]}\n"
                    f"- Key factors: {', '.join(str(k) for k in kf[:5])}\n"
                )

        sections.append(
            "\nBased on the weekend intelligence above, write your weekend "
            "briefing as JSON focusing on what will happen when the market "
            "reopens on Monday."
        )

        prompt = "\n".join(sections)

        try:
            response = self._llm.invoke([
                SystemMessage(content=_WEEKEND_NARRATOR_SYSTEM),
                HumanMessage(content=prompt),
            ])
            result = json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning("Weekend narrator returned invalid JSON – using defaults")
            result = self._weekend_defaults()
        except Exception as e:
            logger.error(f"Weekend narrator LLM call failed: {e}")
            result = self._weekend_defaults()

        # Validate
        outlook = str(result.get("overall_outlook", "neutral")).lower()
        if outlook not in {"bullish", "bearish", "neutral"}:
            outlook = "neutral"
        result["overall_outlook"] = outlook

        result = _sanitize_technical_names(result)
        return result

    @staticmethod
    def _weekend_defaults() -> dict:
        return {
            "overall_outlook": "neutral",
            "executive_summary": (
                "🔍 What's Happening: The gold market is closed for the weekend. "
                "No trading activity is taking place.\n"
                "📊 Why It Matters: Weekend developments could affect Monday's opening price.\n"
                "👀 What to Watch on Monday: Monitor any geopolitical or economic news "
                "that emerged over the weekend."
            ),
            "bull_case": "Not available — weekend analysis could not be generated.",
            "bear_case": "Not available — weekend analysis could not be generated.",
            "risk_factors": [
                "Weekend geopolitical developments could cause a gap at Monday's open"
            ],
        }

    def _build_prompt(
        self,
        ml_predictions: list[dict],
        agent_reports: dict[str, dict],
        shap_explanation: Optional[dict],
        current_price: float,
        feedback_context: str,
    ) -> str:
        try:
            return self._build_prompt_inner(
                ml_predictions, agent_reports, shap_explanation,
                current_price, feedback_context,
            )
        except Exception as e:
            logger.warning(f"Narrator prompt build failed ({e}), using minimal prompt")
            return (
                f"Current Indian gold price: ₹{current_price:,.2f} per 10g.\n"
                f"ML model predicts {len(ml_predictions)} hours.\n"
                "Write your narrative briefing as JSON."
            )

    def _build_prompt_inner(
        self,
        ml_predictions: list[dict],
        agent_reports: dict[str, dict],
        shap_explanation: Optional[dict],
        current_price: float,
        feedback_context: str,
    ) -> str:
        sections = []

        # Current price
        sections.append(f"Current Indian gold price: ₹{current_price:,.2f} per 10g")

        # ML prediction summary
        if ml_predictions:
            first = ml_predictions[0]
            last = ml_predictions[-1]
            sections.append(
                f"\n## ML Ensemble Predictions (next {len(ml_predictions)} hours)\n"
                f"- Hour 1: ₹{first.get('xgb_price', 0):,.2f} "
                f"[₹{first.get('xgb_low', 0):,.2f} – ₹{first.get('xgb_high', 0):,.2f}]\n"
                f"- Hour {len(ml_predictions)}: ₹{last.get('xgb_price', 0):,.2f} "
                f"[₹{last.get('xgb_low', 0):,.2f} – ₹{last.get('xgb_high', 0):,.2f}]\n"
                f"- Direction: {'UP' if last.get('xgb_price', 0) > first.get('xgb_price', 0) else 'DOWN'}\n"
                f"- Model components (Hour 1): "
                f"XGBoost=₹{first.get('component_xgb', 0):,.2f}, "
                f"LightGBM=₹{first.get('component_lgb', 0):,.2f}, "
                f"Ridge=₹{first.get('component_ridge', 0):,.2f}"
            )

        # SHAP explanation
        if shap_explanation:
            top_features = shap_explanation.get("feature_importance", [])[:10]
            if top_features:
                feat_lines = [
                    f"  {f['feature']}: {f['importance']:.4f}"
                    for f in top_features
                ]
                sections.append(
                    "\n## SHAP Feature Importance (top 10 drivers)\n"
                    + "\n".join(feat_lines)
                )

            hourly = shap_explanation.get("hourly_drivers", [])
            if hourly:
                h_lines = [
                    f"  Hour {h['hour']}: {', '.join(str(d) for d in h.get('drivers', []))}"
                    for h in hourly[:6]  # show first 6 hours
                ]
                sections.append(
                    "\n## Per-Hour Key Drivers (from SHAP)\n" + "\n".join(h_lines)
                )

        # Agent summaries (for context, not for price prediction)
        if agent_reports:
            agent_sec = ["\n## Agent Intelligence Reports\n"]
            for name, report in agent_reports.items():
                if not isinstance(report, dict):
                    continue
                bias = report.get('prediction_bias', 0)
                bias = bias if isinstance(bias, (int, float)) else 0
                conf = report.get('confidence', 0)
                conf = conf if isinstance(conf, (int, float)) else 0
                kf = report.get('key_factors', []) or []
                agent_sec.append(
                    f"### {name.replace('_', ' ').title()}\n"
                    f"- Outlook: {report.get('outlook', 'N/A')} | "
                    f"Confidence: {conf:.2f} | "
                    f"Bias: {bias:+.2f}\n"
                    f"- Summary: {str(report.get('summary', ''))[:500]}\n"
                    f"- Key factors: {', '.join(str(k) for k in kf[:5])}\n"
                )
            sections.append("".join(agent_sec))

        # Feedback
        if feedback_context:
            sections.append(f"\n## Recent Accuracy Feedback\n{feedback_context}")

        sections.append(
            "\nBased on all of the above, write your narrative briefing as JSON."
        )

        return "\n".join(sections)

    @staticmethod
    def _defaults() -> dict:
        return {
            "overall_outlook": "neutral",
            "executive_summary": "Analysis could not be generated.",
            "bull_case": "",
            "bear_case": "",
            "risk_factors": [],
            "hourly_drivers": ["Market activity" for _ in range(PREDICTION_HOURS)],
        }


def compute_ml_confidence(predictions: list[dict], current_price: float) -> float:
    """
    Derive overall confidence from ML band widths (NOT from LLM).

    Narrow bands relative to price → high confidence.
    Wide bands → low confidence.
    """
    if not predictions or current_price <= 0:
        return 0.4

    band_pcts = []
    for p in predictions:
        lo = p.get("xgb_low", 0)
        hi = p.get("xgb_high", 0)
        mid = p.get("xgb_price", current_price)
        if mid > 0:
            band_pcts.append((hi - lo) / mid * 100)

    if not band_pcts:
        return 0.4

    avg_band_pct = sum(band_pcts) / len(band_pcts)

    # Map: 0.5% band → 0.85 conf, 1% → 0.75, 2% → 0.60, 4% → 0.40, 8% → 0.20
    if avg_band_pct <= 0.5:
        conf = 0.85
    elif avg_band_pct <= 2.0:
        conf = 0.85 - (avg_band_pct - 0.5) * 0.167  # linear decay
    elif avg_band_pct <= 5.0:
        conf = 0.60 - (avg_band_pct - 2.0) * 0.067
    else:
        conf = max(0.20, 0.40 - (avg_band_pct - 5.0) * 0.04)

    return round(max(0.10, min(0.90, conf)), 2)
