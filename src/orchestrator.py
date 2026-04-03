"""
Orchestrator – runs all specialist agents in parallel, aggregates their reports,
and produces a unified 7-day gold price prediction via a meta-reasoning LLM call.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
from typing import Any

from langchain_openai import AzureChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
from loguru import logger
from pydantic import BaseModel, Field

from .config import (
    AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY,
    AZURE_OPENAI_DEPLOYMENT, AZURE_OPENAI_API_VERSION,
    TEMPERATURE, PREDICTION_HOURS, CACHE_DIR,
)
from .agents.base_agent import AgentReport
from .agents.geopolitics_agent import GeopoliticsAgent
from .agents.trend_analysis_agent import TrendAnalysisAgent
from .agents.etf_flow_agent import ETFFlowAgent
from .agents.macro_economics_agent import MacroEconomicsAgent
from .agents.oil_energy_agent import OilEnergyAgent
from .agents.sentiment_agent import SentimentAgent
from .agents.technical_agent import TechnicalAnalysisAgent
from .agents.historical_pattern_agent import HistoricalPatternAgent
from .data_fetchers.market_data import MarketDataFetcher
from .guardrails import validate_prediction_plan, adjust_confidence_from_track_record
from .time_utils import iso_now_ist, now_ist


# ── Prediction schema ──────────────────────────────────────────────


class DayPrediction(BaseModel):
    date: str
    predicted_price: float
    low_range: float
    high_range: float
    confidence: float
    key_driver: str


class PredictionPlan(BaseModel):
    generated_at: str = Field(default_factory=iso_now_ist)
    current_price: float = 0.0
    overall_outlook: str = "neutral"  # bullish / bearish / neutral
    overall_confidence: float = 0.5
    executive_summary: str = ""
    daily_predictions: list[DayPrediction] = Field(default_factory=list)
    risk_factors: list[str] = Field(default_factory=list)
    bull_case: str = ""
    bear_case: str = ""
    agent_reports: dict[str, dict] = Field(default_factory=dict)  # slim copies


META_SYSTEM_PROMPT = """You are the Chief Gold Strategist for the INDIAN gold market.
You are the most senior analyst in a multi-agent gold prediction system.
All prices are in INR (Indian Rupees) per 10 grams.

You have received independent reports from 8 specialist analysts (geopolitics, trend,
ETF flows, macro-economics, oil/energy, sentiment, technical analysis, historical patterns),
all focused on the Indian gold market.

Your job is to:
1. Weigh each analyst's findings by their confidence AND impact score.
2. Identify consensus and disagreements.
3. Produce a precise hourly rolling prediction with price targets in INR per 10g and confidence bands.
4. Consider the dual drivers: global gold price (COMEX) AND USD/INR exchange rate.
5. Explain the bull case and bear case for Indian gold.
6. List the top risk factors that could invalidate the prediction.

CRITICAL REALISM RULES for hourly predictions:
- Gold does NOT move monotonically. Include realistic pullbacks, consolidations, and
  reversals within the 24h horizon (e.g., a bullish outlook may still have 2-3 dip hours).
- Avoid uniform increments. Real price moves cluster around news events and market opens
  (MCX 9am IST, COMEX 8:20pm IST). Some hours should be flat or slightly contrarian.
- Confidence MUST decay with horizon: hours 1-6 can be 0.70-0.85, hours 7-12 should
  drop to 0.55-0.70, hours 13-24 should be 0.40-0.60. The future is uncertain.
- Widen bands for later hours. Hour 1 band ±₹200-₹400, hour 24 band ±₹600-₹1200.
- Each hour's key_driver should reflect what is ACTUALLY driving that specific hour
  (market session, data release, technical level), not recycle the same 5 reasons.

Return ONLY valid JSON with these EXACT keys (no markdown fences):
{
  "overall_outlook": "bullish" | "bearish" | "neutral",
  "overall_confidence": 0.0 to 1.0,
  "executive_summary": "3-4 paragraph synthesis of all agent findings for Indian gold",
  "daily_predictions": [
    {
      "date": "YYYY-MM-DD HH:00",
      "predicted_price": float (INR per 10g),
      "low_range": float,
      "high_range": float,
      "confidence": 0.0 to 1.0,
      "key_driver": "what drives this hour's move"
    },
    ... (exactly one entry per hour for the next horizon, starting from next full hour)
  ],
  "risk_factors": ["risk 1", "risk 2", ...],
  "bull_case": "paragraph describing the bullish scenario for Indian gold",
  "bear_case": "paragraph describing the bearish scenario for Indian gold"
}
"""


def _extract_summary_text(text: str) -> str:
    """Normalize agent summary text, including partial JSON-like outputs."""
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    if cleaned.startswith("{"):
        try:
            obj = json.loads(cleaned)
            summary = obj.get("summary") if isinstance(obj, dict) else None
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
        except Exception:
            # Best-effort extraction when the JSON is truncated/invalid.
            match = re.search(r'"summary"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned, flags=re.DOTALL)
            if match:
                raw_value = match.group(1)
                try:
                    return bytes(raw_value, "utf-8").decode("unicode_escape").strip()
                except Exception:
                    return raw_value.replace('\\"', '"').strip()

    return cleaned


def _ensure_full_stop(text: str) -> str:
    """Ensure summary text ends cleanly with a period."""
    cleaned = (text or "").strip()
    if not cleaned:
        return ""

    cleaned = cleaned.rstrip(" ")
    if cleaned.endswith("..."):
        return cleaned
    if cleaned.endswith("…"):
        return cleaned.rstrip("… ") + "..."
    if cleaned.endswith("."):
        return cleaned
    if cleaned.endswith("!") or cleaned.endswith("?"):
        return cleaned[:-1].rstrip() + "."
    return cleaned + "."


def _truncate_sentence_safe(text: str, max_chars: int) -> str:
    """Truncate text at a natural boundary so UI copy does not end mid-sentence."""
    if len(text) <= max_chars:
        return _ensure_full_stop(text)

    snippet = text[:max_chars].rstrip()
    cut = max(snippet.rfind(". "), snippet.rfind("! "), snippet.rfind("? "), snippet.rfind("\n"))
    if cut >= int(max_chars * 0.6):
        return _ensure_full_stop(snippet[: cut + 1])

    word_cut = snippet.rfind(" ")
    if word_cut > 0:
        return _ensure_full_stop(snippet[:word_cut])
    return _ensure_full_stop(snippet)


def _format_summary(text: str, max_chars: int) -> str:
    """Normalize, truncate, and ensure a clean sentence-ending summary."""
    normalized = _extract_summary_text(text)
    if not normalized:
        return "No summary available."
    return _truncate_sentence_safe(normalized, max_chars)


def _json_safe(value: Any) -> Any:
    """Best-effort conversion to JSON-serialisable types for plan persistence."""
    if value is None:
        return None
    if isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, dict):
        out: dict[str, Any] = {}
        for k, v in value.items():
            try:
                out[str(k)] = _json_safe(v)
            except Exception:
                out[str(k)] = str(v)
        return out

    # Common non-JSON-friendly numeric types (numpy, pandas scalars, etc.)
    try:
        if hasattr(value, "item"):
            return _json_safe(value.item())
    except Exception:
        pass

    return str(value)


class Orchestrator:
    """Run all agents → aggregate → meta-reason → produce PredictionPlan."""

    AGENT_CLASSES = [
        GeopoliticsAgent,
        TrendAnalysisAgent,
        ETFFlowAgent,
        MacroEconomicsAgent,
        OilEnergyAgent,
        SentimentAgent,
        TechnicalAnalysisAgent,
        HistoricalPatternAgent,
    ]

    def __init__(self):
        self._llm = AzureChatOpenAI(
            azure_endpoint=AZURE_OPENAI_ENDPOINT,
            azure_deployment=AZURE_OPENAI_DEPLOYMENT,
            api_key=AZURE_OPENAI_API_KEY,
            api_version=AZURE_OPENAI_API_VERSION,
            temperature=TEMPERATURE,
            request_timeout=90,
        )
        self._market = MarketDataFetcher()

    # ------------------------------------------------------------------ #
    def run_all_agents(self) -> list[AgentReport]:
        """Execute every specialist agent in parallel threads."""
        reports: list[AgentReport] = []

        with ThreadPoolExecutor(max_workers=8) as pool:
            futures = {
                pool.submit(cls().run): cls.NAME if hasattr(cls, "NAME") else cls.__name__
                for cls in self.AGENT_CLASSES
            }
            for future in as_completed(futures):
                name = futures[future]
                try:
                    report = future.result()
                    reports.append(report)
                except Exception as e:
                    logger.error(f"Agent {name} failed: {e}")
                    reports.append(
                        AgentReport(
                            agent_name=name,
                            summary=f"Agent failed: {e}",
                            outlook="neutral",
                            confidence=0.0,
                        )
                    )

        # Keep report order stable across runs so the meta-prompt is deterministic.
        agent_order = {
            (cls.NAME if hasattr(cls, "NAME") else cls.__name__): i
            for i, cls in enumerate(self.AGENT_CLASSES)
        }
        reports.sort(key=lambda r: agent_order.get(r.agent_name, 999))

        return reports

    # ------------------------------------------------------------------ #
    def _build_meta_prompt(self, reports: list[AgentReport], current_price: float) -> str:
        """Assemble the meta-reasoning prompt from all agent reports."""
        this_hour = now_ist().replace(minute=0, second=0, microsecond=0)
        start_hour = this_hour + timedelta(hours=1)
        dates = [(start_hour + timedelta(hours=i)).strftime("%Y-%m-%d %H:00") for i in range(PREDICTION_HOURS)]

        agent_summaries = []
        for r in reports:
            summary_for_prompt = _format_summary(r.summary, 800)
            agent_summaries.append(
                f"### {r.agent_name}\n"
                f"- Outlook: {r.outlook} | Confidence: {r.confidence:.2f} | "
                f"Impact: {r.impact_score:.2f} | Bias: {r.prediction_bias:+.2f}\n"
                f"- Summary: {summary_for_prompt}\n"
                f"- Key factors: {', '.join(r.key_factors[:5])}\n"
            )

        return f"""Current Indian gold price: \u20b9{current_price:,.2f} per 10 grams
Prediction dates needed: {', '.join(dates)}

## Recent Forecast Error Feedback
{self._build_feedback_context()}

## Agent Reports

{''.join(agent_summaries)}

Synthesise all of the above and produce your hourly prediction plan for Indian gold (INR per 10g) as JSON."""

    def _build_feedback_context(self) -> str:
        """Summarise recent forecast misses so the next run can self-correct."""
        log_path = Path(CACHE_DIR) / "accuracy_log.json"
        if not log_path.exists():
            return "No prior evaluation data available."

        try:
            entries = json.loads(log_path.read_text(encoding="utf-8"))
        except Exception:
            return "Prior evaluation data unreadable."

        if not isinstance(entries, list) or not entries:
            return "No prior evaluation data available."

        recent = entries[-8:]
        signed_errors: list[float] = []
        abs_pct_errors: list[float] = []
        for ev in recent:
            for item in ev.get("daily_results", []) or []:
                try:
                    signed_errors.append(float(item.get("error", 0.0)))
                    abs_pct_errors.append(abs(float(item.get("pct_error", 0.0))))
                except Exception:
                    continue

        if not signed_errors:
            return "No prior evaluation data available."

        mean_signed = sum(signed_errors) / len(signed_errors)
        mean_abs_pct = sum(abs_pct_errors) / len(abs_pct_errors) if abs_pct_errors else 0.0
        tendency = "over-predicting" if mean_signed > 0 else "under-predicting"
        return (
            f"Recent evaluated plans: {len(recent)}. "
            f"Average signed error: {mean_signed:+.2f} INR ({tendency}). "
            f"Average absolute percentage error: {mean_abs_pct:.2f}%. "
            "Use this as calibration feedback for the next-hour forecast."
        )

    # ------------------------------------------------------------------ #
    def _get_recent_mape(self) -> float | None:
        """Read recent MAPE from accuracy log (if available)."""
        log_path = Path(CACHE_DIR) / "accuracy_log.json"
        try:
            if not log_path.exists():
                return None
            entries = json.loads(log_path.read_text(encoding="utf-8"))
            if not isinstance(entries, list) or not entries:
                return None
            recent = entries[-5:]
            mapes = [e.get("aggregate", {}).get("mape") for e in recent if e.get("aggregate")]
            mapes = [m for m in mapes if m is not None]
            return sum(mapes) / len(mapes) if mapes else None
        except Exception:
            return None

    def _get_recent_band_hit_rate(self) -> float | None:
        """Read recent band hit rate from accuracy log (if available)."""
        log_path = Path(CACHE_DIR) / "accuracy_log.json"
        try:
            if not log_path.exists():
                return None
            entries = json.loads(log_path.read_text(encoding="utf-8"))
            if not isinstance(entries, list) or not entries:
                return None
            recent = entries[-5:]
            rates = [e.get("aggregate", {}).get("band_hit_rate") for e in recent if e.get("aggregate")]
            rates = [r for r in rates if r is not None]
            return sum(rates) / len(rates) if rates else None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def generate_prediction(self) -> PredictionPlan:
        """Full pipeline: agents → meta-reasoning → PredictionPlan."""
        logger.info("=== Orchestrator: starting full prediction cycle ===")

        # 1. Get current gold price in INR per 10 grams
        current_price = self._market.get_gold_inr_price()
        if not isinstance(current_price, (int, float)) or not math.isfinite(current_price) or current_price <= 0:
            logger.warning("Invalid INR gold price; attempting fallback via COMEX + USD/INR")
            fallback_df = self._market.fetch_ticker("GC=F", period_days=14)
            usdinr = self._market.get_usdinr_rate()
            if not fallback_df.empty and "Close" in fallback_df:
                close = fallback_df["Close"].squeeze()
                close = close.dropna() if hasattr(close, "dropna") else close
                if len(close) > 0:
                    usd_per_oz = float(close.iloc[-1])
                    current_price = round(usd_per_oz * usdinr / 31.1035 * 10, 2)

        logger.info(f"Current Indian gold price: \u20b9{current_price:,.2f} per 10g")

        # 2. Run all agents
        reports = self.run_all_agents()
        logger.info(f"Received {len(reports)} agent reports")

        # 3. Meta-reasoning
        meta_prompt = self._build_meta_prompt(reports, current_price)
        try:
            response = self._llm.invoke([
                SystemMessage(content=META_SYSTEM_PROMPT),
                HumanMessage(content=meta_prompt),
            ])
            raw = response.content
            result = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Meta-reasoning returned invalid JSON – using defaults")
            result = {
                "overall_outlook": "neutral",
                "overall_confidence": 0.3,
                "executive_summary": raw if 'raw' in dir() else "Analysis unavailable",
                "daily_predictions": [],
                "risk_factors": [],
                "bull_case": "",
                "bear_case": "",
            }
        except Exception as e:
            logger.error(f"Meta-reasoning LLM call failed: {e}")
            result = {
                "overall_outlook": "neutral",
                "overall_confidence": 0.0,
                "executive_summary": f"Meta-reasoning failed: {e}",
                "daily_predictions": [],
                "risk_factors": [],
                "bull_case": "",
                "bear_case": "",
            }

        # 4. ── Guardrail: validate meta-LLM output ──
        result = validate_prediction_plan(result, current_price, PREDICTION_HOURS)

        # 5. ── Guardrail: track-record confidence adjustment ──
        result["overall_confidence"] = adjust_confidence_from_track_record(
            result["overall_confidence"],
            mape=self._get_recent_mape(),
            band_hit_rate=self._get_recent_band_hit_rate(),
        )

        # 6. Build PredictionPlan
        daily = []
        for dp in result.get("daily_predictions", []):
            try:
                daily.append(DayPrediction(**dp))
            except Exception:
                pass

        # Normalise to fixed hourly horizon even when LLM returns fewer/misaligned rows.
        start_hour = now_ist().replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        target_times = [start_hour + timedelta(hours=i) for i in range(PREDICTION_HOURS)]
        normalised: list[DayPrediction] = []
        for i, ts in enumerate(target_times):
            src = daily[i] if i < len(daily) else (daily[-1] if daily else None)
            if src is None:
                base = current_price
                normalised.append(
                    DayPrediction(
                        date=ts.strftime("%Y-%m-%d %H:00"),
                        predicted_price=round(base, 2),
                        low_range=round(base * 0.995, 2),
                        high_range=round(base * 1.005, 2),
                        confidence=0.4,
                        key_driver="Fallback hourly baseline",
                    )
                )
            else:
                normalised.append(
                    DayPrediction(
                        date=ts.strftime("%Y-%m-%d %H:00"),
                        predicted_price=float(src.predicted_price),
                        low_range=float(src.low_range),
                        high_range=float(src.high_range),
                        confidence=float(src.confidence),
                        key_driver=str(src.key_driver),
                    )
                )
        daily = normalised

        plan = PredictionPlan(
            current_price=current_price,
            overall_outlook=result.get("overall_outlook", "neutral"),
            overall_confidence=float(result.get("overall_confidence", 0.5)),
            executive_summary=result.get("executive_summary", ""),
            daily_predictions=daily,
            risk_factors=result.get("risk_factors", []),
            bull_case=result.get("bull_case", ""),
            bear_case=result.get("bear_case", ""),
            agent_reports={
                r.agent_name: {
                    "outlook": r.outlook,
                    "confidence": r.confidence,
                    "impact_score": r.impact_score,
                    "prediction_bias": r.prediction_bias,
                    "summary": _format_summary(r.summary, 700),
                    "key_factors": r.key_factors[:5],
                    "data_points": _json_safe(r.data_points),
                }
                for r in reports
            },
        )

        logger.info(
            f"=== Prediction complete: {plan.overall_outlook} "
            f"(conf={plan.overall_confidence:.2f}) ==="
        )
        return plan
