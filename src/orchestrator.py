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
from .time_utils import iso_now_ist, now_ist, current_slot_ist, SLOT_HOURS, is_market_closed_ist
from .ml_ensemble import MLEnsemble
from .signal_extractor import extract_signals
from .narrator import LLMNarrator, compute_ml_confidence


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


def _extract_summary_text(text: str) -> str:
    """Normalize agent summary text, including partial JSON-like outputs."""
    if not isinstance(text, str):
        return ""

    cleaned = text.strip()
    if not cleaned:
        return ""

    # Strip markdown code fences the LLM sometimes wraps around JSON
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        lines = lines[1:]  # drop opening fence
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]  # drop closing fence
        cleaned = "\n".join(lines).strip()

    if cleaned.startswith("{"):
        try:
            obj = json.loads(cleaned)
            summary = obj.get("summary") if isinstance(obj, dict) else None
            if isinstance(summary, str) and summary.strip():
                return summary.strip()
        except Exception:
            pass

        # Best-effort regex extraction (handles truncated/malformed JSON)
        match = re.search(r'"summary"\s*:\s*"((?:\\.|[^"\\])*)"', cleaned, flags=re.DOTALL)
        if match:
            raw_value = match.group(1)
            try:
                return bytes(raw_value, "utf-8").decode("unicode_escape").strip()
            except Exception:
                return raw_value.replace('\\"', '"').strip()

        # Last resort: strip the JSON wrapper to extract readable content.
        # Handles cases where the closing quote is missing (truncated JSON).
        match = re.search(r'"summary"\s*:\s*"((?:\\.|[^"\\])*)', cleaned, flags=re.DOTALL)
        if match:
            raw_value = match.group(1)
            return raw_value.replace('\\"', '"').replace("\\n", "\n").strip()

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
        self._ml_ensemble = MLEnsemble()
        self._narrator = LLMNarrator()

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
            mapes = [e.get("mape") for e in recent if e.get("mape") is not None]
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
            rates = [e.get("band_hit_rate") for e in recent if e.get("band_hit_rate") is not None]
            return sum(rates) / len(rates) if rates else None
        except Exception:
            return None

    # ------------------------------------------------------------------ #
    def generate_prediction(self) -> PredictionPlan:
        """ML-first pipeline: agents → signals → ML ensemble → narrator → PredictionPlan.

        Architecture:
          1. Get current gold price
          2. Run all 8 specialist agents (intelligence gathering)
          3. Extract numeric signals from agent reports
          4. Train ML ensemble (XGBoost + LightGBM + Ridge stacking)
          5. ML ensemble forecasts next 24 hours (all prices come from ML)
          6. Compute SHAP explainability
          7. LLM narrator writes prose (executive summary, bull/bear, risk factors)
          8. Assemble PredictionPlan

        The LLM NEVER generates price numbers — only narrative.
        """
        logger.info("=== Orchestrator: starting ML-first prediction cycle ===")

        # 1. Get current gold price in INR per 10 grams
        current_price = self._market.get_gold_inr_price()
        if not isinstance(current_price, (int, float)) or not math.isfinite(current_price) or current_price <= 0:
            logger.warning("Invalid INR gold price from MCX; attempting fallback via COMEX + USD/INR")
            fallback_df = self._market.fetch_ticker("GC=F", period_days=14)
            usdinr = self._market.get_usdinr_rate()
            if not fallback_df.empty and "Close" in fallback_df:
                close = fallback_df["Close"].squeeze()
                close = close.dropna() if hasattr(close, "dropna") else close
                if len(close) > 0:
                    usd_per_oz = float(close.iloc[-1])
                    current_price = round(usd_per_oz * usdinr / 31.1035 * 10, 2)

        # Validate that we have a sane INR/10g gold price after all attempts.
        # Typical range ~₹50,000–₹250,000 per 10g as of 2024-2026.
        if (
            not isinstance(current_price, (int, float))
            or not math.isfinite(current_price)
            or current_price <= 0
        ):
            raise ValueError(
                f"Cannot obtain a valid gold price (got {current_price!r}). "
                "Both MCX and COMEX fallback failed."
            )

        logger.info(f"Current Indian gold price: \u20b9{current_price:,.2f} per 10g")

        # 2. Run all agents (intelligence gathering, NOT price prediction)
        reports = self.run_all_agents()
        logger.info(f"Received {len(reports)} agent reports")

        # Build slim agent report dict for plan + narrator
        agent_report_dict = {
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
        }

        # 3. Extract numeric signals from agent reports → ML features
        agent_signals = extract_signals(agent_report_dict)
        logger.info(f"Agent signals: {agent_signals}")

        # 4. Train ML ensemble
        ml_trained = self._ml_ensemble.train(agent_signals)
        if not ml_trained:
            logger.warning("ML ensemble training failed — falling back to basic forecast")

        # 5. ML ensemble predicts next 24 hours
        ml_predictions = []
        if ml_trained:
            ml_predictions = self._ml_ensemble.predict(agent_signals)
            logger.info(f"ML ensemble produced {len(ml_predictions)} hourly predictions")

        # 6. SHAP explainability
        shap_explanation = None
        if ml_trained:
            shap_explanation = self._ml_ensemble.get_shap_explanation(agent_signals)
            if shap_explanation:
                logger.info(
                    f"SHAP: {shap_explanation['total_features']} features, "
                    f"top driver: {shap_explanation['feature_importance'][0]['feature']}"
                )

        # 7. LLM narrator writes prose (NO price generation)
        feedback_ctx = self._build_feedback_context()
        narrative = self._narrator.narrate(
            ml_predictions=ml_predictions,
            agent_reports=agent_report_dict,
            shap_explanation=shap_explanation,
            current_price=current_price,
            feedback_context=feedback_ctx,
        )

        # 8. Assemble PredictionPlan
        #    Prices come from ML, narrative from LLM
        #    Align prediction hours to the 6-hour slot boundary (00, 06, 12, 18 IST)
        #    so different plans generated within the same slot produce the same hour grid.
        slot_start = current_slot_ist().replace(tzinfo=None)
        now = now_ist().replace(tzinfo=None)
        # Start from slot boundary + 1h, but never predict hours already past
        slot_start_hour = slot_start.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        now_start_hour = now.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
        start_hour = max(slot_start_hour, now_start_hour)
        target_times = [start_hour + timedelta(hours=i) for i in range(PREDICTION_HOURS)]
        hourly_drivers = narrative.get("hourly_drivers", [])

        daily: list[DayPrediction] = []
        for i, ts in enumerate(target_times):
            ml_item = ml_predictions[i] if i < len(ml_predictions) else None
            driver = hourly_drivers[i] if i < len(hourly_drivers) else "Market activity"

            if ml_item:
                # Confidence decays with horizon (derived from band width)
                price = float(ml_item.get("xgb_price", current_price))
                lo = float(ml_item.get("xgb_low", price * 0.995))
                hi = float(ml_item.get("xgb_high", price * 1.005))
                band_pct = (hi - lo) / price * 100 if price > 0 else 1.0
                # Tighter bands → higher confidence for that hour
                hour_conf = max(0.20, min(0.90, 0.85 - band_pct * 0.15))
                daily.append(DayPrediction(
                    date=ts.strftime("%Y-%m-%d %H:00"),
                    predicted_price=round(price, 2),
                    low_range=round(lo, 2),
                    high_range=round(hi, 2),
                    confidence=round(hour_conf, 2),
                    key_driver=driver,
                ))
            else:
                # Fallback when ML unavailable: use wider bands with horizon scaling
                band_pct = 0.01 + 0.001 * i  # 1% at h0 → 3.4% at h24
                daily.append(DayPrediction(
                    date=ts.strftime("%Y-%m-%d %H:00"),
                    predicted_price=round(current_price, 2),
                    low_range=round(current_price * (1 - band_pct), 2),
                    high_range=round(current_price * (1 + band_pct), 2),
                    confidence=max(0.2, 0.35 - 0.005 * i),
                    key_driver="Fallback baseline (ML unavailable)",
                ))

        # ── Weekend flatline ────────────────────────────────────────
        # If any predicted hours fall outside MCX trading hours (IST),
        # flatline those hours at the actual current market price with tight bands.
        # MCX Gold is closed on Saturday/Sunday IST and before 09:00 on Monday.
        # Using current_price (the live market price at generation time) ensures
        # weekend predictions reflect the real closing price, not ML extrapolation.
        if daily:
            flatline_price = current_price  # use actual market closing price
            # Pre-parse all dates once to avoid redundant strptime calls
            dp_dates = [
                datetime.strptime(dp.date, "%Y-%m-%d %H:%M") for dp in daily
            ]
            # Flatline all market-closed hours at the actual closing price
            for dp, dp_dt in zip(daily, dp_dates):
                if is_market_closed_ist(dp_dt):
                    dp.predicted_price = round(flatline_price, 2)
                    dp.low_range = round(flatline_price * 0.998, 2)
                    dp.high_range = round(flatline_price * 1.002, 2)
                    dp.confidence = 0.95
                    dp.key_driver = "Market closed (weekend) — price held flat at closing"

        # Overall confidence from ML band widths (NOT from LLM)
        ml_confidence = compute_ml_confidence(ml_predictions, current_price)

        # Track-record adjustment
        _recent_band = self._get_recent_band_hit_rate()
        ml_confidence = adjust_confidence_from_track_record(
            ml_confidence,
            mape=self._get_recent_mape(),
            # band_hit_rate is stored as percentage (e.g. 31.0) in the log
            # but adjust_confidence_from_track_record expects 0-1 fraction
            band_hit_rate=(_recent_band / 100) if _recent_band is not None else None,
        )

        # Store SHAP + component info in agent_reports for UI display
        if shap_explanation:
            agent_report_dict["_ml_ensemble"] = {
                "type": "ml_ensemble",
                "model": "XGBoost + LightGBM + Ridge (stacked)",
                "shap": shap_explanation,
            }

        plan = PredictionPlan(
            current_price=current_price,
            overall_outlook=narrative.get("overall_outlook", "neutral"),
            overall_confidence=float(ml_confidence),
            executive_summary=narrative.get("executive_summary", ""),
            daily_predictions=daily,
            risk_factors=narrative.get("risk_factors", []),
            bull_case=narrative.get("bull_case", ""),
            bear_case=narrative.get("bear_case", ""),
            agent_reports=agent_report_dict,
        )

        # Apply guardrails on the final plan
        plan_dict = json.loads(plan.model_dump_json())
        corrected = validate_prediction_plan(plan_dict, current_price, PREDICTION_HOURS)
        plan = PredictionPlan(**corrected)
        # Restore agent_reports (guardrails don't modify these)
        plan.agent_reports = agent_report_dict

        logger.info(
            f"=== ML-first prediction complete: {plan.overall_outlook} "
            f"(conf={plan.overall_confidence:.2f}, "
            f"ml_preds={len(ml_predictions)}, shap={'yes' if shap_explanation else 'no'}) ==="
        )
        return plan
