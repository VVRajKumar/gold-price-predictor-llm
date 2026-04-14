"""
Residual Learner – online learning from past prediction errors.

After each prediction cycle the accuracy tracker records
(predicted_price, actual_price, signed_error) for every hour.
This module learns multi-dimensional bias patterns and applies corrections
to future ML ensemble predictions so they progressively improve.

Five correction stages (applied in order):
  1. Per-horizon EWMA bias – average signed error for each forecast offset
     (hour 1, hour 2, … hour 24).
  2. Slot-specific bias – predictions generated at 00:00 IST may have
     different systematic errors than those from 18:00 IST, because market
     conditions at generation time affect which data the model sees.
  3. Recent-momentum correction – if the last N overlapping predictions for
     a given hour consistently overshoot (or undershoot), apply a momentum
     correction to the next prediction covering that hour.
  4. Agent-accuracy tracking – track which agent signals historically
     correlate with prediction accuracy and adjust weights dynamically.
  5. Adaptive band adjustment – widen or tighten confidence bands based on
     the historical out-of-band rate per horizon.
"""

from __future__ import annotations

import json
import math
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from loguru import logger

from .config import CACHE_DIR, PREDICTION_HOURS
from .time_utils import SLOT_HOURS

_CORRECTION_CACHE = CACHE_DIR / "residual_corrections.json"

# ── Minimum data thresholds ─────────────────────────────────────────
_MIN_SAMPLES_BIAS = 3       # need ≥3 data points per horizon to start correcting
_MIN_SAMPLES_STRONG = 15    # ramp correction strength up to this
_MAX_CORRECTION_PCT = 1.5   # cap correction at ±1.5% of price

# ── Slot-specific learning ──────────────────────────────────────────
_MIN_SLOT_SAMPLES = 2       # need ≥2 data points per slot+horizon to apply slot correction
_SLOT_CORRECTION_WEIGHT = 0.4  # how much slot-specific bias influences vs global bias

# ── Recent-momentum correction ──────────────────────────────────────
_MOMENTUM_WINDOW = 3        # look at last 3 overlapping predictions for same hour
_MOMENTUM_THRESHOLD = 0.6   # if ≥60% of recent predictions err in same direction, correct
_MOMENTUM_STRENGTH = 0.25   # correction strength for momentum signal
_MAX_MOMENTUM_PCT = 0.5     # cap momentum correction at ±0.5% of price

# ── Agent-accuracy learning ─────────────────────────────────────────
_MIN_AGENT_SAMPLES = 5      # need ≥5 samples to start adjusting agent weights
_AGENT_WEIGHT_RANGE = (0.5, 2.0)  # agent weight multiplier bounds

# Maximum recent-error hour keys to cache (~4 days × 24 hours × 2 slots overlap)
_MAX_RECENT_HOUR_KEYS = 192

# ── Self-healing: over-correction detection (#10) ───────────────────
# If residual corrections worsened accuracy in the recent window, halve
# correction strength to prevent systematic over-correction after regime changes.
_SELF_HEAL_WINDOW = 12      # look at last 12 hours of corrections
_SELF_HEAL_DAMPEN = 0.5     # halve correction strength when over-correcting


class ResidualLearner:
    """Learns from past prediction errors and produces corrections."""

    def __init__(self):
        self._bias: dict[int, float] = {}       # horizon → EWMA of signed error
        self._band_miss: dict[int, float] = {}   # horizon → fraction of band misses
        self._sample_count: dict[int, int] = {}  # horizon → number of observations

        # ── NEW: Slot-specific bias (IST generation-hour → horizon → bias) ──
        self._slot_bias: dict[int, dict[int, float]] = {}  # slot_hour → {horizon → bias}
        self._slot_count: dict[int, dict[int, int]] = {}   # slot_hour → {horizon → count}

        # ── NEW: Recent prediction errors for momentum detection ──
        # Key: predicted hour string → list of recent (signed_error, gen_at) tuples
        self._recent_errors: dict[str, list[tuple[float, str]]] = {}

        # ── NEW: Agent-accuracy weights (learned from error correlation) ──
        # agent_name → weight multiplier (>1 = trust more, <1 = trust less)
        self._agent_weights: dict[str, float] = {}
        # agent_name → list of (signal_value, prediction_error_pct) for correlation
        self._agent_error_history: dict[str, list[tuple[float, float]]] = {}

        # ── Self-healing (#10): track if corrections are helping or hurting ──
        # Stores recent (corrected_error_pct, uncorrected_error_pct) pairs
        self._correction_performance: list[tuple[float, float]] = []
        self._overcorrecting: bool = False

        self._load_cache()

    # ── Persistence ──────────────────────────────────────────────────

    def _load_cache(self):
        if _CORRECTION_CACHE.exists():
            try:
                data = json.loads(_CORRECTION_CACHE.read_text(encoding="utf-8"))
                self._bias = {int(k): v for k, v in data.get("bias", {}).items()}
                self._band_miss = {int(k): v for k, v in data.get("band_miss", {}).items()}
                self._sample_count = {int(k): v for k, v in data.get("sample_count", {}).items()}

                # Load slot-specific bias
                for slot_str, horizon_dict in data.get("slot_bias", {}).items():
                    slot = int(slot_str)
                    self._slot_bias[slot] = {int(k): v for k, v in horizon_dict.items()}
                for slot_str, horizon_dict in data.get("slot_count", {}).items():
                    slot = int(slot_str)
                    self._slot_count[slot] = {int(k): v for k, v in horizon_dict.items()}

                # Load recent errors
                self._recent_errors = data.get("recent_errors", {})

                # Load agent weights
                self._agent_weights = data.get("agent_weights", {})

                # Load self-healing state
                self._correction_performance = data.get("correction_performance", [])
                self._overcorrecting = data.get("overcorrecting", False)

                logger.debug(
                    f"Residual learner loaded: {len(self._bias)} horizons, "
                    f"{len(self._slot_bias)} slots, "
                    f"{len(self._agent_weights)} agent weights"
                )
            except Exception:
                pass

    def _save_cache(self):
        data = {
            "bias": self._bias,
            "band_miss": self._band_miss,
            "sample_count": self._sample_count,
            "slot_bias": {str(k): v for k, v in self._slot_bias.items()},
            "slot_count": {str(k): v for k, v in self._slot_count.items()},
            "recent_errors": self._recent_errors,
            "agent_weights": self._agent_weights,
            "correction_performance": self._correction_performance[-_SELF_HEAL_WINDOW:],
            "overcorrecting": self._overcorrecting,
        }
        try:
            from . import cloud_storage
            cloud_storage.persist("residual_corrections.json", data)
        except Exception as e:
            logger.warning(f"Could not save residual correction cache: {e}")

    # ── Learn from accuracy log ──────────────────────────────────────

    def update_from_accuracy_log(
        self,
        accuracy_log: list[dict],
        agent_signals_history: Optional[dict[str, list[tuple[float, float]]]] = None,
    ):
        """
        Recompute all learned patterns from the full accuracy log.

        Learns:
          1. Per-horizon EWMA bias (global)
          2. Per-slot + per-horizon EWMA bias (00:00 vs 06:00 vs 12:00 vs 18:00)
          3. Recent per-hour momentum (last N overlapping predictions)
          4. Agent-accuracy correlations (if agent_signals_history provided)
          5. Band miss rates per horizon
        """
        # ── Collect errors grouped by (horizon, slot, predicted_hour) ──
        horizon_errors: dict[int, list[float]] = defaultdict(list)
        horizon_misses: dict[int, list[bool]] = defaultdict(list)
        slot_horizon_errors: dict[int, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
        hour_recent: dict[str, list[tuple[float, str]]] = defaultdict(list)

        for evaluation in accuracy_log:
            gen_at = evaluation.get("plan_generated_at", "")
            try:
                gen_dt = datetime.fromisoformat(str(gen_at)).replace(tzinfo=None)
            except (ValueError, TypeError):
                continue

            # Determine which IST slot this plan was generated in
            gen_slot = _nearest_slot(gen_dt.hour)

            for result in evaluation.get("daily_results", []):
                pred_date = result.get("date", "")
                try:
                    pred_dt = datetime.fromisoformat(str(pred_date)).replace(tzinfo=None)
                except (ValueError, TypeError):
                    try:
                        pred_dt = datetime.strptime(str(pred_date), "%Y-%m-%d %H:%M:%S")
                    except (ValueError, TypeError):
                        continue

                delta_hours = round((pred_dt - gen_dt).total_seconds() / 3600)
                if delta_hours < 1 or delta_hours > PREDICTION_HOURS:
                    continue

                signed_error = result.get("error", 0.0)  # predicted - actual
                within_band = result.get("within_band", True)

                # 1. Global per-horizon
                horizon_errors[delta_hours].append(float(signed_error))
                horizon_misses[delta_hours].append(not within_band)

                # 2. Slot-specific per-horizon
                slot_horizon_errors[gen_slot][delta_hours].append(float(signed_error))

                # 3. Recent per-hour (for momentum detection)
                pred_hour_key = pred_dt.strftime("%Y-%m-%d %H:00")
                hour_recent[pred_hour_key].append((float(signed_error), gen_at))

        # ── 1. Compute global EWMA bias per horizon ──
        alpha = 0.3
        new_bias: dict[int, float] = {}
        new_miss: dict[int, float] = {}
        new_count: dict[int, int] = {}

        for h in range(1, PREDICTION_HOURS + 1):
            errors = horizon_errors.get(h, [])
            misses = horizon_misses.get(h, [])
            if not errors:
                continue
            new_count[h] = len(errors)
            ewma = errors[0]
            for e in errors[1:]:
                ewma = alpha * e + (1.0 - alpha) * ewma
            new_bias[h] = ewma
            new_miss[h] = sum(1 for m in misses if m) / len(misses) if misses else 0.0

        self._bias = new_bias
        self._band_miss = new_miss
        self._sample_count = new_count

        # ── 2. Compute slot-specific EWMA bias ──
        new_slot_bias: dict[int, dict[int, float]] = {}
        new_slot_count: dict[int, dict[int, int]] = {}

        for slot in SLOT_HOURS:
            slot_errors = slot_horizon_errors.get(slot, {})
            if not slot_errors:
                continue
            new_slot_bias[slot] = {}
            new_slot_count[slot] = {}
            for h in range(1, PREDICTION_HOURS + 1):
                errors = slot_errors.get(h, [])
                if not errors:
                    continue
                new_slot_count.setdefault(slot, {})[h] = len(errors)
                ewma = errors[0]
                for e in errors[1:]:
                    ewma = alpha * e + (1.0 - alpha) * ewma
                new_slot_bias.setdefault(slot, {})[h] = ewma

        self._slot_bias = new_slot_bias
        self._slot_count = new_slot_count

        # ── 3. Store recent momentum data (last N per hour) ──
        new_recent: dict[str, list[tuple[float, str]]] = {}
        for hour_key, error_list in hour_recent.items():
            # Keep only the most recent _MOMENTUM_WINDOW entries
            sorted_entries = sorted(error_list, key=lambda x: x[1], reverse=True)
            new_recent[hour_key] = sorted_entries[:_MOMENTUM_WINDOW]
        # Only keep recent hours (limit cache size to ~4 days of data)
        if new_recent:
            all_keys = sorted(new_recent.keys(), reverse=True)
            self._recent_errors = {k: new_recent[k] for k in all_keys[:_MAX_RECENT_HOUR_KEYS]}
        else:
            self._recent_errors = {}

        # ── 4. Update agent-accuracy weights ──
        if agent_signals_history:
            self._update_agent_weights(agent_signals_history)

        # ── 5. Self-healing: assess if corrections are helping (#10) ──
        self._assess_correction_performance(accuracy_log)

        self._save_cache()

        if new_bias:
            avg_bias = np.mean(list(new_bias.values()))
            avg_miss = np.mean(list(new_miss.values())) * 100
            total = sum(new_count.values())
            slot_info = ", ".join(
                f"slot {s}h: {len(b)} horizons"
                for s, b in new_slot_bias.items()
            )
            logger.info(
                f"Residual learner updated: {len(new_bias)} horizons, "
                f"{total} total samples, avg bias ₹{avg_bias:+,.0f}, "
                f"band miss {avg_miss:.0f}%, "
                f"slots: [{slot_info}], "
                f"momentum keys: {len(self._recent_errors)}, "
                f"agent weights: {len(self._agent_weights)}"
            )

    def _update_agent_weights(
        self, agent_signals_history: dict[str, list[tuple[float, float]]]
    ):
        """Learn which agents produce signals that correlate with accurate predictions.

        For each agent, we track (signal_value, pct_error) pairs.  Agents whose
        signals correlate with lower error get higher weight multipliers; those
        whose signals correlate with higher error get penalized.

        The correlation is simple: when the agent's signal magnitude is high and
        the prediction error is low, the agent is helping.  When signal magnitude
        is high but error is also high, the agent may be misleading.
        """
        for agent_name, pairs in agent_signals_history.items():
            if len(pairs) < _MIN_AGENT_SAMPLES:
                continue

            # Split into high-signal and low-signal groups
            abs_signals = [abs(s) for s, _ in pairs]
            median_signal = float(np.median(abs_signals)) if abs_signals else 0.0

            high_signal_errors = [abs(e) for s, e in pairs if abs(s) > median_signal]
            low_signal_errors = [abs(e) for s, e in pairs if abs(s) <= median_signal]

            if not high_signal_errors or not low_signal_errors:
                continue

            avg_high = np.mean(high_signal_errors)
            avg_low = np.mean(low_signal_errors)

            # If the agent's strong signals lead to LOWER error than weak signals,
            # it's adding value → increase its weight.
            # If strong signals lead to HIGHER error, it's misleading → decrease weight.
            if avg_low > 0:
                ratio = avg_high / avg_low
                if ratio < 0.85:
                    # Strong signals → lower error: agent is helping
                    weight = min(_AGENT_WEIGHT_RANGE[1], 1.0 + (1.0 - ratio) * 0.5)
                elif ratio > 1.15:
                    # Strong signals → higher error: agent is misleading
                    weight = max(_AGENT_WEIGHT_RANGE[0], 1.0 - (ratio - 1.0) * 0.3)
                else:
                    weight = 1.0
                self._agent_weights[agent_name] = round(weight, 3)

        if self._agent_weights:
            logger.info(
                f"Agent weight adjustments: "
                + ", ".join(f"{k}={v:.2f}" for k, v in self._agent_weights.items())
            )

    # ── Self-healing: assess correction performance (#10) ──────────

    def _assess_correction_performance(self, accuracy_log: list[dict]):
        """Assess whether residual corrections improved or worsened accuracy.

        Compares the actual signed error against the bias correction that was
        applied.  If corrections have been systematically making predictions
        *worse* over the recent window, set ``_overcorrecting = True`` so that
        ``correct_predictions`` halves correction strength.
        """
        if not self._bias:
            self._overcorrecting = False
            return

        # We look at the most recent evaluations (up to _SELF_HEAL_WINDOW plans)
        recent = accuracy_log[-_SELF_HEAL_WINDOW:] if accuracy_log else []
        if not recent:
            return

        helped = 0
        hurt = 0
        for evaluation in recent:
            try:
                gen_dt = datetime.fromisoformat(
                    str(evaluation.get("plan_generated_at", ""))
                ).replace(tzinfo=None)
            except (ValueError, TypeError):
                continue

            for result in evaluation.get("daily_results", []):
                try:
                    pred_dt = datetime.fromisoformat(
                        str(result.get("date", ""))
                    ).replace(tzinfo=None)
                except (ValueError, TypeError):
                    try:
                        pred_dt = datetime.strptime(
                            str(result.get("date", "")), "%Y-%m-%d %H:%M:%S"
                        )
                    except (ValueError, TypeError):
                        continue

                delta_hours = round((pred_dt - gen_dt).total_seconds() / 3600)
                if delta_hours < 1 or delta_hours > PREDICTION_HOURS:
                    continue

                signed_error = result.get("error", 0.0)
                bias_correction = self._bias.get(delta_hours, 0.0)

                # If the correction was in the opposite direction of the error,
                # it helped (reduced the error).  If same direction, it hurt.
                if bias_correction != 0 and signed_error != 0:
                    if (signed_error > 0 and bias_correction > 0) or \
                       (signed_error < 0 and bias_correction < 0):
                        # Correction direction matches error → it helped
                        helped += 1
                    else:
                        # Correction direction opposes error → it may have hurt
                        hurt += 1

        total = helped + hurt
        if total < 6:
            # Not enough data to judge
            return

        hurt_ratio = hurt / total
        was_overcorrecting = self._overcorrecting

        # If >60% of corrections worsened accuracy, flag as over-correcting
        self._overcorrecting = hurt_ratio > 0.6

        if self._overcorrecting and not was_overcorrecting:
            logger.warning(
                f"[residual:self-heal] Over-correction detected: "
                f"{hurt}/{total} corrections worsened accuracy → halving strength"
            )
        elif not self._overcorrecting and was_overcorrecting:
            logger.info(
                f"[residual:self-heal] Over-correction resolved: "
                f"{helped}/{total} corrections now helping → restoring full strength"
            )

    # ── Apply corrections ────────────────────────────────────────────

    def correct_predictions(
        self,
        xgb_preds: list[dict],
        generation_slot: Optional[int] = None,
    ) -> list[dict]:
        """
        Apply learned residual corrections to ML ensemble predictions.

        Five-stage correction:
          1. Global per-horizon bias (subtract EWMA of signed errors)
          2. Slot-specific bias (if we know which IST slot generated this prediction)
          3. Recent-momentum correction (if overlapping predictions consistently err)
          4. [Agent weights applied externally via get_agent_weight_adjustments()]
          5. Band widening for high miss-rate horizons

        Returns a new list of corrected prediction dicts.
        """
        if not self._bias and not self._slot_bias:
            return xgb_preds  # no learned data yet

        corrected = []
        for i, pred in enumerate(xgb_preds):
            horizon = i + 1
            price = pred.get("xgb_price", 0.0)
            low = pred.get("xgb_low", 0.0)
            high = pred.get("xgb_high", 0.0)

            n_samples = self._sample_count.get(horizon, 0)

            # ── Stage 1: Global per-horizon bias correction ──
            global_bias = self._bias.get(horizon, 0.0)

            # ── Stage 2: Slot-specific bias correction ──
            slot_bias = 0.0
            slot_samples = 0
            if generation_slot is not None and generation_slot in self._slot_bias:
                slot_bias = self._slot_bias[generation_slot].get(horizon, 0.0)
                slot_samples = self._slot_count.get(generation_slot, {}).get(horizon, 0)

            # Blend global and slot-specific bias
            if slot_samples >= _MIN_SLOT_SAMPLES and n_samples >= _MIN_SAMPLES_BIAS:
                # Slot-specific bias is more targeted; blend with global
                combined_bias = (
                    (1 - _SLOT_CORRECTION_WEIGHT) * global_bias
                    + _SLOT_CORRECTION_WEIGHT * slot_bias
                )
            else:
                combined_bias = global_bias

            if n_samples >= _MIN_SAMPLES_BIAS and price > 0:
                # Ramp correction strength from 0.3 to 0.8 based on sample count
                strength = min(
                    0.3 + 0.5 * (n_samples - _MIN_SAMPLES_BIAS)
                    / max(_MIN_SAMPLES_STRONG - _MIN_SAMPLES_BIAS, 1),
                    0.8,
                )

                # Self-healing: halve correction strength if over-correcting (#10)
                if self._overcorrecting:
                    strength *= _SELF_HEAL_DAMPEN

                max_corr = price * (_MAX_CORRECTION_PCT / 100.0)
                clamped_bias = max(-max_corr, min(combined_bias, max_corr))

                correction = clamped_bias * strength
                price -= correction
                low -= correction
                high -= correction

            # ── Stage 3: Recent-momentum correction ──
            pred_hour_key = pred.get("date", "")
            momentum_corr = self._compute_momentum_correction(pred_hour_key, price)
            if momentum_corr != 0.0:
                price -= momentum_corr
                low -= momentum_corr
                high -= momentum_corr

            # ── Stage 5: Band widening for high miss-rate horizons ──
            miss_rate = self._band_miss.get(horizon, 0.0)
            if miss_rate > 0.2 and n_samples >= _MIN_SAMPLES_BIAS:
                band_half = (high - low) / 2.0
                widen_factor = 1.0 + 1.25 * (miss_rate - 0.2)
                new_half = band_half * widen_factor
                if price > 0:
                    max_half = price * 0.04
                    new_half = min(new_half, max_half)
                mid = (high + low) / 2.0
                low = mid - new_half
                high = mid + new_half

            corrected.append({
                "date": pred["date"],
                "xgb_price": round(price, 2),
                "xgb_low": round(low, 2),
                "xgb_high": round(high, 2),
            })

        return corrected

    def _compute_momentum_correction(self, pred_hour_key: str, price: float) -> float:
        """Compute momentum correction for a specific predicted hour.

        If recent overlapping predictions for this hour consistently erred
        in the same direction, apply a correction to counter that trend.
        """
        recent = self._recent_errors.get(pred_hour_key, [])
        if len(recent) < 2:
            return 0.0

        errors = [e for e, _ in recent]
        positive_count = sum(1 for e in errors if e > 0)
        negative_count = sum(1 for e in errors if e < 0)
        total = len(errors)

        # Check if errors are consistently in one direction
        if positive_count / total >= _MOMENTUM_THRESHOLD:
            # Consistently overshooting → correct downward
            avg_error = np.mean([e for e in errors if e > 0])
            correction = avg_error * _MOMENTUM_STRENGTH
        elif negative_count / total >= _MOMENTUM_THRESHOLD:
            # Consistently undershooting → correct upward
            avg_error = np.mean([e for e in errors if e < 0])
            correction = avg_error * _MOMENTUM_STRENGTH
        else:
            return 0.0

        # Cap at _MAX_MOMENTUM_PCT of price
        if price > 0:
            max_mom = price * (_MAX_MOMENTUM_PCT / 100.0)
            correction = max(-max_mom, min(correction, max_mom))

        return correction

    # ── Agent weight adjustments (for external use) ──────────────────

    def get_agent_weight_adjustments(self) -> dict[str, float]:
        """Return learned agent weight multipliers.

        Returns a dict of agent_name → weight_multiplier.
        Values > 1.0 mean the agent has been historically helpful.
        Values < 1.0 mean the agent has been misleading.
        Missing agents default to 1.0 (no adjustment).
        """
        return dict(self._agent_weights)

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_correction_summary(self) -> Optional[dict]:
        """Return a summary of learned corrections for display/logging."""
        if not self._bias and not self._slot_bias:
            return None

        summary: dict = {
            "horizons_learned": len(self._bias),
            "total_samples": sum(self._sample_count.values()),
            "avg_bias_inr": round(float(np.mean(list(self._bias.values()))), 2) if self._bias else 0,
            "max_bias_inr": round(float(max(self._bias.values(), key=abs)), 2) if self._bias else 0,
            "avg_band_miss_pct": round(
                float(np.mean(list(self._band_miss.values()))) * 100, 1
            ) if self._band_miss else 0,
            "per_horizon": {
                h: {
                    "bias": round(self._bias.get(h, 0), 2),
                    "miss_rate": round(self._band_miss.get(h, 0) * 100, 1),
                    "samples": self._sample_count.get(h, 0),
                }
                for h in sorted(self._bias.keys())
            },
        }

        # Slot-specific summary
        if self._slot_bias:
            summary["slot_corrections"] = {
                f"{slot}:00": {
                    "horizons": len(biases),
                    "avg_bias": round(float(np.mean(list(biases.values()))), 2) if biases else 0,
                    "samples": sum(self._slot_count.get(slot, {}).values()),
                }
                for slot, biases in self._slot_bias.items()
            }

        # Momentum summary
        active_momentum = sum(
            1 for recent in self._recent_errors.values()
            if len(recent) >= 2
        )
        summary["momentum_active_hours"] = active_momentum

        # Agent weight summary
        if self._agent_weights:
            summary["agent_weight_adjustments"] = dict(self._agent_weights)

        return summary


def _nearest_slot(hour: int) -> int:
    """Map an hour (0-23) to the nearest IST slot (0, 6, 12, 18).

    Hours before the first slot (e.g. 0-5) map to slot 0.
    """
    for slot in sorted(SLOT_HOURS, reverse=True):
        if hour >= slot:
            return slot
    return SLOT_HOURS[0]  # hours before first slot → slot 0
