"""
Signal Extractor – converts LLM agent reports into numeric features for ML.

Each agent already returns structured fields (outlook, confidence, impact_score,
prediction_bias, plus domain-specific keys like fear_greed_level, rsi signal, etc.).
This module maps those into the 8 floating-point signals consumed by the ML ensemble:

    sentiment_score      (-1 to +1)   from SentimentAgent
    geopolitical_risk    ( 0 to  1)   from GeopoliticsAgent
    macro_outlook        (-1 to +1)   from MacroEconomicsAgent
    technical_signal     (-1 to +1)   from TechnicalAnalysisAgent
    etf_flow_signal      (-1 to +1)   from ETFFlowAgent
    oil_energy_signal    (-1 to +1)   from OilEnergyAgent
    historical_seasonal  (-1 to +1)   from HistoricalPatternAgent
    trend_strength       (-1 to +1)   from TrendAnalysisAgent

The extraction relies ONLY on the structured keys already returned by agents —
no additional LLM calls are needed.  If a field is missing we fall back to the
generic (outlook, bias, confidence) triple.
"""

from __future__ import annotations
from typing import Any

from loguru import logger


# ── Lookup tables for categorical → numeric ─────────────────────────

_OUTLOOK_MAP = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}

_FEAR_GREED_MAP = {
    "extreme_fear": -1.0,
    "fear": -0.5,
    "neutral": 0.0,
    "greed": 0.5,
    "extreme_greed": 1.0,
}

_INSTITUTIONAL_DEMAND_MAP = {
    "strong_buying": 1.0,
    "buying": 0.5,
    "neutral": 0.0,
    "selling": -0.5,
    "strong_selling": -1.0,
}

_RBI_OUTLOOK_MAP = {
    "cutting": 0.5,    # rate cuts → weaker INR → gold +
    "holding": 0.0,
    "hiking": -0.5,    # rate hikes → stronger INR → gold −
}

_INFLATION_MAP = {
    "rising": 0.5,     # inflation → gold +
    "stable": 0.0,
    "falling": -0.5,
}

_INR_OUTLOOK_MAP = {
    "weakening": 0.5,  # weaker INR → higher Indian gold price
    "stable": 0.0,
    "strengthening": -0.5,
}

_ENERGY_RISK_MAP = {
    "high": 0.5,
    "moderate": 0.0,
    "low": -0.3,
}

_SEASONAL_MAP = _OUTLOOK_MAP  # same mapping

_TREND_STRENGTH_MAP = {
    "strong": 1.0,
    "moderate": 0.5,
    "weak": 0.2,
}

_RSI_MAP = {
    "overbought": -0.5,  # overbought → potential reversal down
    "neutral": 0.0,
    "oversold": 0.5,     # oversold → potential bounce up
}

_MACD_MAP = _OUTLOOK_MAP


def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        v = float(val)
        return v if -10.0 <= v <= 10.0 else default
    except (TypeError, ValueError):
        return default


def _generic_signal(report: dict) -> float:
    """Fallback: derive a -1..+1 signal from (outlook, bias, confidence)."""
    outlook = _OUTLOOK_MAP.get(str(report.get("outlook", "neutral")).lower(), 0.0)
    bias = _safe_float(report.get("prediction_bias"), 0.0)
    conf = _safe_float(report.get("confidence"), 0.5)
    # Weighted combo: bias carries the most information
    return max(-1.0, min(1.0, 0.3 * outlook + 0.5 * bias + 0.2 * (conf - 0.5) * 2))


def extract_signals(agent_reports: dict[str, dict]) -> dict[str, float]:
    """
    Convert the plan's agent_reports dict into 8 numeric features.

    Parameters
    ----------
    agent_reports : dict[str, dict]
        Keyed by agent name (e.g. "sentiment_agent"), values are the
        report dicts stored in PredictionPlan.agent_reports.

    Returns
    -------
    dict with exactly 8 float keys matching AGENT_SIGNAL_NAMES in ml_ensemble.py.
    """
    signals: dict[str, float] = {}

    # ── Sentiment ────────────────────────────────────────────────
    sent = agent_reports.get("sentiment_agent", {})
    if sent:
        news_score = _safe_float(sent.get("data_points", {}).get("news_sentiment_score"))
        fg = _FEAR_GREED_MAP.get(
            str(sent.get("data_points", {}).get("fear_greed_level", "neutral")).lower(), 0.0
        )
        if news_score != 0.0:
            signals["sentiment_score"] = max(-1, min(1, 0.6 * news_score + 0.4 * fg))
        else:
            signals["sentiment_score"] = _generic_signal(sent)
    else:
        signals["sentiment_score"] = 0.0

    # ── Geopolitics ──────────────────────────────────────────────
    geo = agent_reports.get("geopolitics_agent", {})
    if geo:
        # Impact score (0-1) already captures how much geopolitics matters right now
        impact = _safe_float(geo.get("impact_score"), 0.5)
        outlook = _OUTLOOK_MAP.get(str(geo.get("outlook", "neutral")).lower(), 0.0)
        bias = _safe_float(geo.get("prediction_bias"), 0.0)
        # geopolitical_risk: 0 = calm, 1 = high tension (gold-supportive)
        # bullish outlook + positive bias → elevated risk → safe-haven gold demand
        # neutral → moderate risk, bearish → reduced risk signal
        direction_factor = 0.5 + 0.25 * outlook + 0.25 * bias
        signals["geopolitical_risk"] = max(0.0, min(1.0, impact * direction_factor))
    else:
        signals["geopolitical_risk"] = 0.3  # mild baseline

    # ── Macro Economics ──────────────────────────────────────────
    macro = agent_reports.get("macro_economics_agent", {})
    if macro:
        dp = macro.get("data_points", {})
        rbi = _RBI_OUTLOOK_MAP.get(str(dp.get("rbi_rate_outlook", "holding")).lower(), 0.0)
        infl = _INFLATION_MAP.get(str(dp.get("inflation_trend", "stable")).lower(), 0.0)
        inr = _INR_OUTLOOK_MAP.get(str(dp.get("inr_outlook", "stable")).lower(), 0.0)
        bias = _safe_float(macro.get("prediction_bias"), 0.0)
        # Blend structured data (RBI + inflation + INR) with LLM's prediction bias
        structured = rbi + infl + inr
        signals["macro_outlook"] = max(-1, min(1, 0.6 * structured + 0.4 * bias))
    else:
        signals["macro_outlook"] = _generic_signal(macro) if macro else 0.0

    # ── Technical Analysis ───────────────────────────────────────
    tech = agent_reports.get("technical_analysis_agent", {})
    if tech:
        dp = tech.get("data_points", {})
        sigs = dp.get("signals", {}) if isinstance(dp.get("signals"), dict) else {}
        rsi_val = _RSI_MAP.get(str(sigs.get("rsi", "neutral")).lower(), 0.0)
        macd_val = _MACD_MAP.get(str(sigs.get("macd", "neutral")).lower(), 0.0)
        bias = _safe_float(tech.get("prediction_bias"), 0.0)
        signals["technical_signal"] = max(-1, min(1, 0.3 * rsi_val + 0.3 * macd_val + 0.4 * bias))
    else:
        signals["technical_signal"] = 0.0

    # ── ETF Flows ────────────────────────────────────────────────
    etf = agent_reports.get("etf_flow_agent", {})
    if etf:
        dp = etf.get("data_points", {})
        inst = _INSTITUTIONAL_DEMAND_MAP.get(
            str(dp.get("institutional_demand", "neutral")).lower(), 0.0
        )
        bias = _safe_float(etf.get("prediction_bias"), 0.0)
        signals["etf_flow_signal"] = max(-1, min(1, 0.5 * inst + 0.5 * bias))
    else:
        signals["etf_flow_signal"] = 0.0

    # ── Oil / Energy ─────────────────────────────────────────────
    oil = agent_reports.get("oil_energy_agent", {})
    if oil:
        dp = oil.get("data_points", {})
        energy_risk = _ENERGY_RISK_MAP.get(
            str(dp.get("energy_inflation_risk", "moderate")).lower(), 0.0
        )
        bias = _safe_float(oil.get("prediction_bias"), 0.0)
        signals["oil_energy_signal"] = max(-1, min(1, 0.4 * energy_risk + 0.6 * bias))
    else:
        signals["oil_energy_signal"] = 0.0

    # ── Historical Patterns ──────────────────────────────────────
    hist = agent_reports.get("historical_pattern_agent", {})
    if hist:
        dp = hist.get("data_points", {})
        seasonal = _SEASONAL_MAP.get(
            str(dp.get("seasonal_bias", "neutral")).lower(), 0.0
        )
        bias = _safe_float(hist.get("prediction_bias"), 0.0)
        signals["historical_seasonal"] = max(-1, min(1, 0.5 * seasonal + 0.5 * bias))
    else:
        signals["historical_seasonal"] = 0.0

    # ── Trend Analysis ───────────────────────────────────────────
    trend = agent_reports.get("trend_analysis_agent", {})
    if trend:
        dp = trend.get("data_points", {})
        strength = _TREND_STRENGTH_MAP.get(
            str(dp.get("trend_strength", "moderate")).lower(), 0.5
        )
        outlook_val = _OUTLOOK_MAP.get(str(trend.get("outlook", "neutral")).lower(), 0.0)
        bias = _safe_float(trend.get("prediction_bias"), 0.0)
        # Combine: direction (outlook) × magnitude (strength), blended with bias
        signals["trend_strength"] = max(-1, min(1, 0.6 * (outlook_val * strength) + 0.4 * bias))
    else:
        signals["trend_strength"] = 0.0

    logger.debug(f"Extracted agent signals: {signals}")
    return signals
