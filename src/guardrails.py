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

# ── Default (static) caps ───────────────────────────────────────────
# These are floors — dynamic caps can raise them based on recent volatility.
_DEFAULT_MAX_HOURLY_MOVE_PCT = 1.0
_DEFAULT_MAX_TOTAL_DEVIATION_PCT = 5.0

# Maximum hourly price change (%) considered plausible.
# Gold can move ~1% in a single hour during volatile sessions; allow room
# for genuine moves while still catching obvious errors.
_MAX_HOURLY_MOVE_PCT = _DEFAULT_MAX_HOURLY_MOVE_PCT

# Maximum total deviation (%) from current price over the full 24-hour horizon.
# Indian gold can move 5%+ during volatile days (geopolitical shocks, rate
# decisions).  A tighter cap forces predictions to under-shoot, inflating MAPE.
_MAX_TOTAL_DEVIATION_PCT = _DEFAULT_MAX_TOTAL_DEVIATION_PCT

# Band width limits (as % of predicted price)
# _MIN_BAND_PCT is now horizon-aware: see _min_band_pct(h)
_MIN_BAND_PCT_BASE = 0.5   # minimum band at hour 0 (near-term)
_MIN_BAND_PCT_SLOPE = 0.03  # additional band per hour
_MAX_BAND_PCT = 5.0         # band cannot be wider than 5% of price


def _min_band_pct(horizon_idx: int) -> float:
    """Horizon-aware minimum band width (%).

    Near-term (h=0): 0.5% → tighter bands reward precision.
    Far-horizon (h=23): 1.19% → wider bands honestly reflect uncertainty.
    """
    return _MIN_BAND_PCT_BASE + _MIN_BAND_PCT_SLOPE * horizon_idx


# Horizon-aware band deviation envelope (shared formula used by guardrails).
# Returns the max allowed deviation (as a fraction) from predicted price for bands.
def _band_envelope_pct(horizon_idx: int) -> float:
    """Max band deviation at a given horizon step (fraction, e.g. 0.02 = 2%)."""
    return min(0.08, 0.015 + 0.003 * horizon_idx)


# ── Per-agent overconfidence thresholds ─────────────────────────────
# Different agent types warrant different confidence limits:
#  - Technical/macro agents rely on quantitative signals → higher threshold OK
#  - News/geopolitics agents rely on ambiguous text → penalize earlier
#  - ETF/sentiment/others → middle ground
_AGENT_OVERCONFIDENCE: dict[str, tuple[float, float]] = {
    # agent_name → (threshold, penalty)
    "technical_analysis_agent": (0.95, 0.10),
    "macro_economics_agent":    (0.95, 0.10),
    "geopolitics_agent":        (0.88, 0.18),
    "sentiment_agent":          (0.90, 0.15),
    "etf_flow_agent":           (0.90, 0.15),
    "oil_energy_agent":         (0.90, 0.15),
    "trend_analysis_agent":     (0.92, 0.12),
    "historical_pattern_agent": (0.92, 0.12),
}

# Fallback for agents not in the dict above
_OVERCONFIDENCE_THRESHOLD = 0.92   # individual agents
_META_OVERCONFIDENCE_THRESHOLD = 0.88   # overall plan
_CONFIDENCE_PENALTY = 0.15          # how much to penalise

# Bias-outlook alignment: treat |bias| below this as "LLM didn't provide a real value"
_BIAS_NEAR_ZERO = 0.05

# ── Monotonic drift detection ──────────────────────────────────────
_MONOTONIC_THRESHOLD = 18   # if > this many of 24 hours drift same direction
_MONOTONIC_DAMPEN = 0.4     # dampen later-hour moves by this factor


# ── Price Validation ────────────────────────────────────────────────

def is_valid_inr_price(price: object) -> bool:
    """Return True if *price* is a finite number in a plausible INR/10g range.

    Valid range: ₹30,000–₹500,000 per 10g.  Rejects NaN, None, non-numeric,
    and values in USD-scale (e.g. ₹3,878) or absurdly high outliers.
    """
    if not isinstance(price, (int, float)):
        return False
    if not math.isfinite(price):
        return False
    return _MIN_INR_PRICE <= price <= _MAX_INR_PRICE


# ── Agent Report Guardrails ─────────────────────────────────────────

def validate_agent_report(report_dict: dict[str, Any], agent_name: str) -> dict[str, Any]:
    """Validate and correct an agent's parsed JSON output before building AgentReport.

    Enforces:
      - outlook ∈ {bullish, bearish, neutral}
      - confidence ∈ [0, 1]
      - impact_score ∈ [0, 1]
      - prediction_bias ∈ [-1, +1]
      - per-agent overconfidence penalty
      - dynamic bias-outlook inference (proportional to impact_score)

    Returns corrected dict (never raises).
    """
    corrections: list[str] = []

    # ── Outlook ──
    outlook = str(report_dict.get("outlook", "neutral")).lower().strip()
    if outlook not in _VALID_OUTLOOKS:
        corrections.append(f"outlook '{outlook}' → 'neutral'")
        outlook = "neutral"
    report_dict["outlook"] = outlook

    # ── Confidence (per-agent overconfidence) ──
    confidence = _safe_float(report_dict.get("confidence"), 0.5)
    confidence = _clamp(confidence, 0.0, 1.0)
    original_conf = confidence

    # Use per-agent thresholds if available, otherwise fall back to global
    agent_threshold, agent_penalty = _AGENT_OVERCONFIDENCE.get(
        agent_name, (_OVERCONFIDENCE_THRESHOLD, _CONFIDENCE_PENALTY)
    )
    if confidence > agent_threshold:
        confidence = max(0.0, confidence - agent_penalty)
        corrections.append(
            f"confidence {original_conf:.2f} → {confidence:.2f} "
            f"(overconfidence: threshold={agent_threshold}, penalty={agent_penalty})"
        )
    report_dict["confidence"] = round(confidence, 3)

    # ── Impact Score ──
    impact = _safe_float(report_dict.get("impact_score"), 0.5)
    impact = _clamp(impact, 0.0, 1.0)
    # Floor: directional outlook with meaningful confidence should have
    # a minimum impact score so the agent actually influences the forecast.
    if outlook in ("bullish", "bearish") and confidence >= 0.3:
        min_impact = round(confidence * 0.4, 3)
        if impact < min_impact:
            corrections.append(
                f"impact_score {impact:.2f} too low for {outlook} outlook "
                f"(conf={confidence:.2f}) → raised to {min_impact:.2f}"
            )
            impact = min_impact
    report_dict["impact_score"] = round(impact, 3)

    # ── Prediction Bias ──
    bias = _safe_float(report_dict.get("prediction_bias"), 0.0)
    bias = _clamp(bias, -1.0, 1.0)
    report_dict["prediction_bias"] = round(bias, 3)

    # ── Bias-Outlook alignment (dynamic inference factor) ──
    # When the outlook is directional but bias is missing / near-zero,
    # infer a meaningful bias from the confidence and impact score so
    # that the agent's opinion actually influences the ML ensemble.
    # The inference factor scales with impact: high-impact agents get
    # a stronger inferred bias.
    _inference_factor = min(0.8, impact)
    if outlook == "bullish" and bias < _BIAS_NEAR_ZERO:
        inferred = round(confidence * _inference_factor, 3)
        if bias < -0.3:
            corrections.append(
                f"bias {bias:+.2f} contradicts bullish outlook → inferred +{inferred:.2f}"
            )
        else:
            corrections.append(
                f"bias {bias:+.2f} too low for bullish outlook → inferred +{inferred:.2f}"
            )
        report_dict["prediction_bias"] = inferred
    elif outlook == "bearish" and bias > -_BIAS_NEAR_ZERO:
        inferred = round(-confidence * _inference_factor, 3)
        if bias > 0.3:
            corrections.append(
                f"bias {bias:+.2f} contradicts bearish outlook → inferred {inferred:+.2f}"
            )
        else:
            corrections.append(
                f"bias {bias:+.2f} too high for bearish outlook → inferred {inferred:+.2f}"
            )
        report_dict["prediction_bias"] = inferred

    if corrections:
        logger.warning(f"[guardrail:{agent_name}] Corrections: {'; '.join(corrections)}")

    return report_dict


# ── Prediction Plan Guardrails ──────────────────────────────────────

def validate_prediction_plan(
    result: dict[str, Any],
    current_price: float,
    n_hours: int,
    *,
    track_record_penalty: float = 0.0,
    recent_hourly_vol: float | None = None,
    recent_daily_vol: float | None = None,
) -> dict[str, Any]:
    """Validate the meta-LLM JSON response and correct illogical predictions.

    Enforces:
      - current_price is in valid INR/10g range [30K, 500K]
      - overall_outlook is valid
      - overall_confidence clamped & penalised if overconfident
      - every daily_prediction has valid price / range / confidence
      - price is within global INR bounds
      - sequential hour-to-hour moves don't exceed max threshold
      - low ≤ predicted ≤ high
      - band widths within reasonable limits (horizon-aware)
      - band envelope clamp on LLM bands (same as XGBoost gets)
      - monotonic drift dampening (if >18/24 same-direction moves)
      - per-hour track-record confidence penalty (50% of plan penalty)
      - dynamic caps based on recent volatility (if provided)

    Raises ValueError if current_price is outside plausible INR range.

    Returns the corrected dict with an extra key ``_guardrail_correction_count``
    indicating how many corrections were applied.
    """
    # ── Reject invalid anchor price early ──
    if not is_valid_inr_price(current_price):
        raise ValueError(
            f"validate_prediction_plan: current_price ₹{current_price!r} is outside "
            f"valid INR/10g range [₹{_MIN_INR_PRICE:,.0f}–₹{_MAX_INR_PRICE:,.0f}]"
        )

    corrections: list[str] = []

    # ── Volatility-aware dynamic caps (#2) ──
    max_hourly = _MAX_HOURLY_MOVE_PCT
    max_total = _MAX_TOTAL_DEVIATION_PCT
    if recent_hourly_vol is not None and recent_hourly_vol > 0:
        max_hourly = max(_DEFAULT_MAX_HOURLY_MOVE_PCT, 3.0 * recent_hourly_vol)
    if recent_daily_vol is not None and recent_daily_vol > 0:
        max_total = max(_DEFAULT_MAX_TOTAL_DEVIATION_PCT, 4.0 * recent_daily_vol)

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

    # Track-record penalty applied to per-hour confidence (50% of plan penalty)
    per_hour_track_penalty = track_record_penalty * 0.5

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

        # ── Total deviation from current price (catch runaway predictions) ──
        if current_price > 0:
            total_dev_pct = abs(pred - current_price) / current_price * 100
            if total_dev_pct > max_total:
                direction = 1 if pred > current_price else -1
                capped = current_price * (1 + direction * max_total / 100)
                corrections.append(
                    f"hour {i}: total deviation {total_dev_pct:.1f}% from current price "
                    f"exceeds {max_total:.1f}% cap (₹{pred:,.0f} → ₹{capped:,.0f})"
                )
                pred = round(capped, 2)

        # ── Hourly move cap ──
        if prev_price > 0:
            move_pct = abs(pred - prev_price) / prev_price * 100
            if move_pct > max_hourly:
                direction = 1 if pred > prev_price else -1
                capped = prev_price * (1 + direction * max_hourly / 100)
                corrections.append(
                    f"hour {i}: move {move_pct:.1f}% exceeds {max_hourly:.1f}% cap "
                    f"(₹{pred:,.0f} → ₹{capped:,.0f})"
                )
                pred = round(capped, 2)

        # ── Band ordering: low ≤ pred ≤ high ──
        if low > pred:
            low = pred
        if high < pred:
            high = pred

        # ── Band width limits (horizon-aware) ──
        band_width = high - low
        min_band = pred * _min_band_pct(i) / 100
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

        # ── Band envelope clamp on LLM plan (#6) ──
        # Apply the same horizon-aware deviation envelope used for XGBoost.
        if current_price > 0:
            max_dev_pct = _band_envelope_pct(i)
            max_dev_abs = current_price * max_dev_pct
            new_low = max(low, pred - max_dev_abs)
            new_high = min(high, pred + max_dev_abs)
            if new_low != low or new_high != high:
                corrections.append(
                    f"hour {i}: band envelope clamped "
                    f"(±{max_dev_pct*100:.1f}% of anchor)"
                )
                low = round(new_low, 2)
                high = round(new_high, 2)

        # ── Confidence decay: later hours → lower confidence ──
        # Hours 1-12: no decay, 13-18: mild decay, 19-24: moderate decay
        horizon_decay = 0.0
        if i >= 18:
            horizon_decay = 0.015 * (i - 17)  # e.g. hour 24 → 0.09
        elif i >= 12:
            horizon_decay = 0.008 * (i - 11)   # e.g. hour 18 → 0.056
        dp_conf = max(0.30, dp_conf - horizon_decay)

        # ── Overconfidence for individual hours ──
        if dp_conf > _OVERCONFIDENCE_THRESHOLD:
            dp_conf = dp_conf - _CONFIDENCE_PENALTY

        # ── Track-record penalty applied to per-hour confidence (#5) ──
        if per_hour_track_penalty > 0:
            dp_conf = max(0.30, dp_conf - per_hour_track_penalty)

        # ── Band widening over horizon ──
        # ML ensemble already applies progressive sqrt(h) widening.
        # Guardrails only enforce minimum/maximum band limits (no additional
        # multiplicative widening to prevent double-stacking).
        # The min/max enforcement above is sufficient.

        dp["predicted_price"] = round(pred, 2)
        dp["low_range"] = round(low, 2)
        dp["high_range"] = round(high, 2)
        dp["confidence"] = round(dp_conf, 3)
        validated_preds.append(dp)

        prev_price = pred

    # ── Monotonic drift detection (#4) ──
    # If >18 of 24 hours all move in the same direction, dampen later moves.
    if len(validated_preds) >= 6:
        directions = []
        pp = current_price
        for dp in validated_preds:
            price = dp["predicted_price"]
            if price > pp:
                directions.append(1)
            elif price < pp:
                directions.append(-1)
            else:
                directions.append(0)
            pp = price

        up_count = sum(1 for d in directions if d > 0)
        down_count = sum(1 for d in directions if d < 0)

        if up_count > _MONOTONIC_THRESHOLD or down_count > _MONOTONIC_THRESHOLD:
            dominant = 1 if up_count > down_count else -1
            corrections.append(
                f"monotonic drift: {up_count} up / {down_count} down → dampening late hours"
            )
            # Dampen moves in the dominant direction for hours after the midpoint
            midpoint = len(validated_preds) // 2
            pp = (
                validated_preds[midpoint - 1]["predicted_price"]
                if midpoint > 0
                else current_price
            )
            for j in range(midpoint, len(validated_preds)):
                dp = validated_preds[j]
                price = dp["predicted_price"]
                move = price - pp
                if (dominant > 0 and move > 0) or (dominant < 0 and move < 0):
                    dampened = pp + move * (1.0 - _MONOTONIC_DAMPEN)
                    dp["predicted_price"] = round(dampened, 2)
                    # Re-centre bands around dampened price
                    band_half = (dp["high_range"] - dp["low_range"]) / 2
                    dp["low_range"] = round(dampened - band_half, 2)
                    dp["high_range"] = round(dampened + band_half, 2)
                    price = dampened
                pp = price

    result["daily_predictions"] = validated_preds

    # ── Outlook-direction consistency (#8) ──
    # If the 24-hour target (last predicted price) clearly contradicts the
    # stated outlook, correct the outlook to match the predicted direction.
    # A small dead-zone (0.1% of current price) avoids flipping on noise.
    if validated_preds and current_price > 0:
        final_price = validated_preds[-1]["predicted_price"]
        price_delta = final_price - current_price
        dead_zone = current_price * 0.001  # 0.1% threshold

        if outlook == "bullish" and price_delta < -dead_zone:
            corrected_outlook = "bearish"
            corrections.append(
                f"overall_outlook 'bullish' contradicts 24h target "
                f"(₹{final_price:,.0f} < ₹{current_price:,.0f}, "
                f"Δ={price_delta:+,.0f}) → '{corrected_outlook}'"
            )
            result["overall_outlook"] = corrected_outlook
        elif outlook == "bearish" and price_delta > dead_zone:
            corrected_outlook = "bullish"
            corrections.append(
                f"overall_outlook 'bearish' contradicts 24h target "
                f"(₹{final_price:,.0f} > ₹{current_price:,.0f}, "
                f"Δ={price_delta:+,.0f}) → '{corrected_outlook}'"
            )
            result["overall_outlook"] = corrected_outlook

    # ── Guardrail correction count (#7) ──
    result["_guardrail_correction_count"] = len(corrections)

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

    If less than 50% of required keys are present the agent is considered
    "data-starved" and the returned warnings include a special marker
    ``_stale_data_fallback`` that callers can use to cap confidence / impact.
    """
    warnings: list[str] = []
    present_count = 0
    for key in required_keys:
        val = data.get(key)
        if val is None:
            warnings.append(f"Missing key '{key}'")
        elif isinstance(val, dict) and not val:
            warnings.append(f"Empty dict for '{key}'")
        elif isinstance(val, (list, str)) and not val:
            warnings.append(f"Empty value for '{key}'")
        else:
            present_count += 1

    is_valid = present_count >= 1  # at least 1 key present

    # ── Stale data hard fallback (#9) ──
    # If fewer than 50% of required keys are available, mark as data-starved
    # so the caller can cap confidence / impact.
    if required_keys and present_count < len(required_keys) * 0.5:
        warnings.append("_stale_data_fallback")
        logger.warning(
            f"[guardrail:data:{agent_name}] Only {present_count}/{len(required_keys)} "
            f"required keys present — flagging for confidence/impact ceiling"
        )

    if warnings:
        logger.warning(f"[guardrail:data:{agent_name}] {'; '.join(w for w in warnings if not w.startswith('_'))}")
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
    """Sanity-check XGBoost predictions before blending with LLM forecast."""
    if not is_valid_inr_price(current_price_inr):
        logger.warning(
            f"validate_xgb_predictions: current_price_inr ₹{current_price_inr!r} "
            f"outside valid INR range — returning empty predictions"
        )
        return []
    validated: list[dict] = []
    prev = current_price_inr

    for i, p in enumerate(preds):
        price = _safe_float(p.get("xgb_price"), prev)
        low = _safe_float(p.get("xgb_low"), price * 0.995)
        high = _safe_float(p.get("xgb_high"), price * 1.005)

        if not math.isfinite(price) or price <= 0:
            price = prev

        # Cap hourly move
        if prev > 0:
            move_pct = abs(price - prev) / prev * 100
            if move_pct > _MAX_HOURLY_MOVE_PCT:
                direction = 1 if price > prev else -1
                price = round(prev * (1 + direction * _MAX_HOURLY_MOVE_PCT / 100), 2)

        # Cap total deviation from current price (prevent compounding drift)
        if current_price_inr > 0:
            total_dev = abs(price - current_price_inr) / current_price_inr * 100
            if total_dev > _MAX_TOTAL_DEVIATION_PCT:
                direction = 1 if price > current_price_inr else -1
                price = round(
                    current_price_inr * (1 + direction * _MAX_TOTAL_DEVIATION_PCT / 100), 2
                )

        # Ensure band ordering
        if low > price:
            low = price
        if high < price:
            high = price

        # Also clamp bands to horizon-aware deviation envelope
        # centred on the *predicted* price so bands stay symmetric.
        if current_price_inr > 0:
            max_dev_pct = _band_envelope_pct(i)
            max_dev_abs = current_price_inr * max_dev_pct
            low = max(low, price - max_dev_abs)
            high = min(high, price + max_dev_abs)

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
) -> tuple[float, float]:
    """Penalise plan confidence when historical accuracy is poor.

    - MAPE > 5%  → mild penalty
    - MAPE > 8%  → heavy penalty
    - band_hit_rate < 20% → additional penalty

    Returns (adjusted_confidence, penalty) — the penalty is also needed by
    ``validate_prediction_plan`` so it can apply 50% of it to per-hour
    confidence values.
    """
    if mape is None and band_hit_rate is None:
        return confidence, 0.0   # no history yet

    penalty = 0.0

    if mape is not None:
        if mape > 8.0:
            penalty += 0.15
        elif mape > 5.0:
            penalty += 0.08

    if band_hit_rate is not None:
        if band_hit_rate < 0.20:
            penalty += 0.10
        elif band_hit_rate < 0.35:
            penalty += 0.05

    adjusted = max(confidence - penalty, 0.30)
    if penalty > 0:
        logger.info(
            f"[guardrail:track-record] Confidence {confidence:.2f} → {adjusted:.2f} "
            f"(MAPE={mape}, band_hit={band_hit_rate})"
        )
    return round(adjusted, 3), round(penalty, 3)


# ── Internal helpers ────────────────────────────────────────────────

def _safe_float(val: Any, default: float = 0.0) -> float:
    try:
        f = float(val)
        return f if math.isfinite(f) else default
    except (TypeError, ValueError):
        return default


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(v, hi))


# ── Volatility-aware dynamic caps (#2) ─────────────────────────────

def compute_dynamic_caps(
    hourly_returns: list[float] | None = None,
) -> tuple[float | None, float | None]:
    """Compute volatility-aware hourly / total deviation caps.

    Parameters
    ----------
    hourly_returns : list of hourly percentage returns from recent COMEX data
                     (e.g. 24-48 hours of hourly pct changes).

    Returns
    -------
    (recent_hourly_vol, recent_daily_vol)
    Either can be None if not enough data.
    """
    if not hourly_returns or len(hourly_returns) < 6:
        return None, None

    import numpy as _np

    valid = [r for r in hourly_returns if math.isfinite(r)]
    if len(valid) < 6:
        return None, None

    hourly_vol = float(_np.std(valid))  # std of hourly pct returns

    # Approximate daily vol as hourly vol × sqrt(24)
    daily_vol = hourly_vol * math.sqrt(24)

    return round(hourly_vol, 4), round(daily_vol, 4)
