"""
Advanced Guardrails – output format enforcement, logical sanity checks,
data integrity validation, and overconfidence penalties.

Every guardrail returns the *corrected* value (clamped, adjusted, or flagged)
so callers can simply replace the original with the return value.
"""

from __future__ import annotations

import math
from typing import Any

from loguru import logger

# ── Boundaries ──────────────────────────────────────────────────────

_VALID_OUTLOOKS = {"bullish", "bearish", "neutral"}

# Indian gold 10g price boundaries (INR).
# As of 2024-2026, gold is ~₹60,000-₹100,000 per 10g.
# We allow a generous range for future fluctuations.
_MIN_INR_PRICE = 30_000.0
_MAX_INR_PRICE = 500_000.0

# Maximum hourly price change (%) considered plausible.
# Even in extreme events, gold rarely moves >5% in a single hour.
_MAX_HOURLY_MOVE_PCT = 5.0
# Public alias so other modules can reference the same constant.
MAX_HOURLY_MOVE_PCT = _MAX_HOURLY_MOVE_PCT

# Maximum total drift from the current actual price across the whole
# forecast horizon.  The per-step cap can compound to very large values
# over many steps; this hard limit anchors every prediction to within
# ±12% of the price at plan-generation time.
_MAX_TOTAL_DRIFT_PCT = 12.0
_MIN_BAND_PCT = 0.1     # band cannot be tighter than 0.1% of price
_MAX_BAND_PCT = 8.0      # band cannot be wider than 8% of price

# Overconfidence thresholds
_OVERCONFIDENCE_THRESHOLD = 0.92   # individual agents
_META_OVERCONFIDENCE_THRESHOLD = 0.88   # overall plan
_CONFIDENCE_PENALTY = 0.15          # how much to penalise


# ── Agent Report Guardrails ─────────────────────────────────────────

def validate_agent_report(report_dict: dict[str, Any], agent_name: str) -> dict[str, Any]:
    """Validate and correct an agent's parsed JSON output before building AgentReport.

    Enforces:
      - outlook ∈ {bullish, bearish, neutral}
      - confidence ∈ [0, 1]
      - impact_score ∈ [0, 1]
      - prediction_bias ∈ [-1, +1]
      - overconfidence penalty

    Returns corrected dict (never raises).
    """
    corrections: list[str] = []

    # ── Outlook ──
    outlook = str(report_dict.get("outlook", "neutral")).lower().strip()
    if outlook not in _VALID_OUTLOOKS:
        corrections.append(f"outlook '{outlook}' → 'neutral'")
        outlook = "neutral"
    report_dict["outlook"] = outlook

    # ── Confidence ──
    confidence = _safe_float(report_dict.get("confidence"), 0.5)
    confidence = _clamp(confidence, 0.0, 1.0)
    original_conf = confidence
    if confidence > _OVERCONFIDENCE_THRESHOLD:
        confidence = confidence - _CONFIDENCE_PENALTY
        corrections.append(
            f"confidence {original_conf:.2f} → {confidence:.2f} (overconfidence penalty)"
        )
    report_dict["confidence"] = round(confidence, 3)

    # ── Impact Score ──
    impact = _safe_float(report_dict.get("impact_score"), 0.5)
    impact = _clamp(impact, 0.0, 1.0)
    report_dict["impact_score"] = round(impact, 3)

    # ── Prediction Bias ──
    bias = _safe_float(report_dict.get("prediction_bias"), 0.0)
    bias = _clamp(bias, -1.0, 1.0)
    report_dict["prediction_bias"] = round(bias, 3)

    # ── Bias-Outlook coherence ──
    if outlook == "bullish" and bias < -0.3:
        corrections.append(f"bias {bias:+.2f} contradicts bullish outlook → clamped to 0.0")
        report_dict["prediction_bias"] = 0.0
    elif outlook == "bearish" and bias > 0.3:
        corrections.append(f"bias {bias:+.2f} contradicts bearish outlook → clamped to 0.0")
        report_dict["prediction_bias"] = 0.0

    if corrections:
        logger.warning(f"[guardrail:{agent_name}] Corrections: {'; '.join(corrections)}")

    return report_dict


# ── Prediction Plan Guardrails ──────────────────────────────────────

def validate_prediction_plan(
    result: dict[str, Any],
    current_price: float,
    n_hours: int,
) -> dict[str, Any]:
    """Validate the meta-LLM JSON response and correct illogical predictions.

    Enforces:
      - overall_outlook is valid
      - overall_confidence clamped & penalised if overconfident
      - every daily_prediction has valid price / range / confidence
      - price is within global INR bounds
      - sequential hour-to-hour moves don't exceed max threshold
      - low ≤ predicted ≤ high
      - band widths within reasonable limits
    """
    corrections: list[str] = []

    # ── Overall fields ──
    outlook = str(result.get("overall_outlook", "neutral")).lower().strip()
    if outlook not in _VALID_OUTLOOKS:
        corrections.append(f"overall_outlook '{outlook}' → 'neutral'")
        outlook = "neutral"
    result["overall_outlook"] = outlook

    conf = _safe_float(result.get("overall_confidence"), 0.5)
    conf = _clamp(conf, 0.0, 1.0)
    if conf > _META_OVERCONFIDENCE_THRESHOLD:
        old = conf
        conf -= _CONFIDENCE_PENALTY
        corrections.append(f"overall_confidence {old:.2f} → {conf:.2f} (overconfidence penalty)")
    result["overall_confidence"] = round(conf, 3)

    # ── Daily predictions ──
    raw_preds = result.get("daily_predictions", [])
    if not isinstance(raw_preds, list):
        raw_preds = []
    validated_preds: list[dict] = []
    prev_price = current_price

    for i, dp in enumerate(raw_preds):
        if not isinstance(dp, dict):
            continue

        pred = _safe_float(dp.get("predicted_price"), prev_price)
        low = _safe_float(dp.get("low_range"), pred * 0.995)
        high = _safe_float(dp.get("high_range"), pred * 1.005)
        dp_conf = _safe_float(dp.get("confidence"), 0.5)
        dp_conf = _clamp(dp_conf, 0.0, 1.0)

        # ── Absolute bounds ──
        if pred < _MIN_INR_PRICE or pred > _MAX_INR_PRICE:
            corrections.append(f"hour {i}: price ₹{pred:,.0f} outside bounds → clamped")
            pred = _clamp(pred, _MIN_INR_PRICE, _MAX_INR_PRICE)

        # ── Hourly move cap (step-to-step) ──
        if prev_price > 0:
            move_pct = abs(pred - prev_price) / prev_price * 100
            if move_pct > _MAX_HOURLY_MOVE_PCT:
                direction = 1 if pred > prev_price else -1
                capped = prev_price * (1 + direction * _MAX_HOURLY_MOVE_PCT / 100)
                corrections.append(
                    f"hour {i}: move {move_pct:.1f}% exceeds {_MAX_HOURLY_MOVE_PCT}% cap "
                    f"(₹{pred:,.0f} → ₹{capped:,.0f})"
                )
                pred = round(capped, 2)

        # ── Absolute drift cap (anchored to current_price) ──
        # The per-step cap can compound over 24 steps; this hard limit
        # prevents exponential drift from the actual current price.
        if current_price > 0:
            total_drift_pct = abs(pred - current_price) / current_price * 100
            if total_drift_pct > _MAX_TOTAL_DRIFT_PCT:
                direction = 1 if pred > current_price else -1
                capped_total = round(
                    current_price * (1 + direction * _MAX_TOTAL_DRIFT_PCT / 100), 2
                )
                corrections.append(
                    f"hour {i}: total drift {total_drift_pct:.1f}% from current "
                    f"₹{current_price:,.0f} → ₹{capped_total:,.0f}"
                )
                pred = capped_total

        # ── Band ordering: low ≤ pred ≤ high ──
        if low > pred:
            low = pred
        if high < pred:
            high = pred

        # ── Band width limits ──
        band_width = high - low
        min_band = pred * _MIN_BAND_PCT / 100
        max_band = pred * _MAX_BAND_PCT / 100

        if band_width < min_band:
            half = min_band / 2
            low = round(pred - half, 2)
            high = round(pred + half, 2)
            corrections.append(f"hour {i}: band too narrow → expanded to ±{min_band/2:.0f}")
        elif band_width > max_band:
            half = max_band / 2
            low = round(pred - half, 2)
            high = round(pred + half, 2)
            corrections.append(f"hour {i}: band too wide → capped to ±{max_band/2:.0f}")

        # ── Confidence decay: later hours → lower confidence ──
        # Hours 1-6: no decay, 7-12: mild decay, 13-24: stronger decay
        horizon_decay = 0.0
        if i >= 12:
            horizon_decay = 0.03 * (i - 11)  # e.g. hour 24 → 0.39
        elif i >= 6:
            horizon_decay = 0.015 * (i - 5)   # e.g. hour 12 → 0.105
        dp_conf = min(dp_conf, max(0.3, dp_conf - horizon_decay))

        # ── Overconfidence for individual hours ──
        if dp_conf > _OVERCONFIDENCE_THRESHOLD:
            dp_conf = dp_conf - _CONFIDENCE_PENALTY

        # ── Band widening over horizon ──
        # Uncertainty grows with forecast distance: widen bands for later hours
        horizon_widen = 1.0 + 0.04 * i  # hour 0→1.0x, hour 12→1.48x, hour 24→1.96x

        dp["predicted_price"] = round(pred, 2)
        dp["low_range"] = round(pred - (pred - low) * horizon_widen, 2)
        dp["high_range"] = round(pred + (high - pred) * horizon_widen, 2)
        dp["confidence"] = round(dp_conf, 3)
        validated_preds.append(dp)

        prev_price = pred

    result["daily_predictions"] = validated_preds

    if corrections:
        logger.warning(f"[guardrail:plan] {len(corrections)} correction(s): {'; '.join(corrections[:10])}")
        if len(corrections) > 10:
            logger.warning(f"[guardrail:plan] ... and {len(corrections) - 10} more")

    return result


# ── Data Integrity Checks ──────────────────────────────────────────

def check_data_freshness(
    data: dict[str, Any],
    required_keys: list[str],
    agent_name: str,
) -> tuple[bool, list[str]]:
    """Verify that gathered data has the necessary keys and non-empty values.

    Returns (is_valid, list_of_warnings).
    """
    warnings: list[str] = []
    for key in required_keys:
        val = data.get(key)
        if val is None:
            warnings.append(f"Missing key '{key}'")
        elif isinstance(val, dict) and not val:
            warnings.append(f"Empty dict for '{key}'")
        elif isinstance(val, (list, str)) and not val:
            warnings.append(f"Empty value for '{key}'")

    is_valid = len(warnings) < len(required_keys)  # at least 1 key present
    if warnings:
        logger.warning(f"[guardrail:data:{agent_name}] {'; '.join(warnings)}")
    return is_valid, warnings


def validate_price_series(values: list[float], label: str) -> list[float]:
    """Remove NaN/Inf and check for suspicious zeros in a price series."""
    cleaned = []
    dropped = 0
    for v in values:
        if not math.isfinite(v) or v <= 0:
            dropped += 1
        else:
            cleaned.append(v)
    if dropped:
        logger.warning(f"[guardrail:series:{label}] Dropped {dropped}/{len(values)} invalid values")
    return cleaned


# ── XGBoost Blend Guardrails ──────────────────────────────────────

def validate_xgb_predictions(
    preds: list[dict],
    current_price_inr: float,
) -> list[dict]:
    """Sanity-check XGBoost predictions before blending with LLM forecast.

    Two layers of protection:
    1. Per-step cap: each predicted price may not move more than
       _MAX_HOURLY_MOVE_PCT% from the previous predicted price.
    2. Absolute-drift cap: the predicted price may not deviate more than
       _MAX_TOTAL_DRIFT_PCT% from `current_price_inr`, regardless of how
       many compounding steps have occurred.  This stops the sequential
       per-step cap from allowing exponential drift over a 24-hour horizon.
    """
    validated: list[dict] = []
    prev = current_price_inr

    for i, p in enumerate(preds):
        price = _safe_float(p.get("xgb_price"), prev)
        low = _safe_float(p.get("xgb_low"), price * 0.995)
        high = _safe_float(p.get("xgb_high"), price * 1.005)

        if not math.isfinite(price) or price <= 0:
            price = prev

        # ── Layer 1: per-step cap (step-to-step) ──
        if prev > 0:
            move_pct = abs(price - prev) / prev * 100
            if move_pct > _MAX_HOURLY_MOVE_PCT:
                direction = 1 if price > prev else -1
                price = round(prev * (1 + direction * _MAX_HOURLY_MOVE_PCT / 100), 2)

        # ── Layer 2: absolute drift cap (anchored to current_price_inr) ──
        if current_price_inr > 0:
            total_drift_pct = abs(price - current_price_inr) / current_price_inr * 100
            if total_drift_pct > _MAX_TOTAL_DRIFT_PCT:
                direction = 1 if price > current_price_inr else -1
                capped = round(
                    current_price_inr * (1 + direction * _MAX_TOTAL_DRIFT_PCT / 100), 2
                )
                logger.debug(
                    f"[guardrail:xgb] step {i}: total drift {total_drift_pct:.1f}% "
                    f"from current ₹{current_price_inr:,.0f} → capped "
                    f"₹{price:,.0f} → ₹{capped:,.0f}"
                )
                price = capped

        # Ensure band ordering
        if low > price:
            low = price
        if high < price:
            high = price

        p["xgb_price"] = round(price, 2)
        p["xgb_low"] = round(low, 2)
        p["xgb_high"] = round(high, 2)
        validated.append(p)
        prev = price

    return validated


# ── Accuracy-Aware Confidence Adjustment ───────────────────────────

def adjust_confidence_from_track_record(
    confidence: float,
    mape: float | None,
    band_hit_rate: float | None,
) -> float:
    """Penalise plan confidence when historical accuracy is poor.

    - MAPE > 2%  → mild penalty
    - MAPE > 5%  → heavy penalty
    - band_hit_rate < 40% → additional penalty
    """
    if mape is None and band_hit_rate is None:
        return confidence   # no history yet

    penalty = 0.0

    if mape is not None:
        if mape > 5.0:
            penalty += 0.20
        elif mape > 2.0:
            penalty += 0.10

    if band_hit_rate is not None:
        if band_hit_rate < 0.30:
            penalty += 0.15
        elif band_hit_rate < 0.50:
            penalty += 0.08

    adjusted = max(confidence - penalty, 0.05)
    if penalty > 0:
        logger.info(
            f"[guardrail:track-record] Confidence {confidence:.2f} → {adjusted:.2f} "
            f"(MAPE={mape}, band_hit={band_hit_rate})"
        )
    return round(adjusted, 3)


# ── Internal helpers ────────────────────────────────────────────────

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))
