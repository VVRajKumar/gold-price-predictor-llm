"""
Residual Learner – online learning from past prediction errors.

After each prediction cycle the accuracy tracker records
(predicted_price, actual_price, signed_error) for every hour.
This module learns per-horizon bias patterns and applies corrections
to future XGBoost predictions so they progressively improve.

Two correction stages:
  1. Per-horizon EWMA bias – learn the average signed error for each
     forecast offset (hour 1, hour 2, … hour 24) and subtract it.
  2. Adaptive band adjustment – widen or tighten confidence bands
     based on the historical out-of-band rate per horizon.
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

_CORRECTION_CACHE = CACHE_DIR / "residual_corrections.json"

# ── Minimum data thresholds ─────────────────────────────────────────
_MIN_SAMPLES_BIAS = 3       # need ≥3 data points per horizon to start correcting
_MIN_SAMPLES_STRONG = 15    # ramp correction strength up to this
_MAX_CORRECTION_PCT = 2.0   # cap correction at ±2% of price (safety limit)


class ResidualLearner:
    """Learns from past prediction errors and produces corrections."""

    def __init__(self):
        self._bias: dict[int, float] = {}       # horizon → EWMA of signed error
        self._band_miss: dict[int, float] = {}   # horizon → fraction of band misses
        self._sample_count: dict[int, int] = {}  # horizon → number of observations
        self._load_cache()

    # ── Persistence ──────────────────────────────────────────────────

    def _load_cache(self):
        if _CORRECTION_CACHE.exists():
            try:
                data = json.loads(_CORRECTION_CACHE.read_text(encoding="utf-8"))
                self._bias = {int(k): v for k, v in data.get("bias", {}).items()}
                self._band_miss = {int(k): v for k, v in data.get("band_miss", {}).items()}
                self._sample_count = {int(k): v for k, v in data.get("sample_count", {}).items()}
                logger.debug(f"Residual learner loaded {len(self._bias)} horizon corrections")
            except Exception:
                pass

    def _save_cache(self):
        data = {
            "bias": self._bias,
            "band_miss": self._band_miss,
            "sample_count": self._sample_count,
        }
        try:
            content = json.dumps(data, indent=2, default=str)
            _CORRECTION_CACHE.write_text(content, encoding="utf-8")
            # Sync to cloud (Gist) so corrections survive Cloud restarts
            from . import cloud_storage
            cloud_storage.persist("residual_corrections.json", content)
        except Exception as e:
            logger.warning(f"Could not save residual correction cache: {e}")

    # ── Learn from accuracy log ──────────────────────────────────────

    def update_from_accuracy_log(self, accuracy_log: list[dict]):
        """
        Recompute per-horizon bias and band-miss stats from the full
        accuracy log (list of evaluation dicts from AccuracyTracker).
        Uses exponentially weighted moving average so recent errors
        matter more than old ones.
        """
        # Collect errors grouped by horizon offset
        horizon_errors: dict[int, list[float]] = defaultdict(list)
        horizon_misses: dict[int, list[bool]] = defaultdict(list)

        for evaluation in accuracy_log:
            gen_at = evaluation.get("plan_generated_at", "")
            try:
                gen_dt = datetime.fromisoformat(str(gen_at)).replace(tzinfo=None)
            except (ValueError, TypeError):
                continue

            for result in evaluation.get("daily_results", []):
                pred_date = result.get("date", "")
                try:
                    pred_dt = datetime.fromisoformat(str(pred_date))
                except (ValueError, TypeError):
                    continue

                # Compute forecast horizon: how many hours ahead was this?
                delta_hours = round((pred_dt - gen_dt).total_seconds() / 3600)
                if delta_hours < 1 or delta_hours > PREDICTION_HOURS:
                    continue

                signed_error = result.get("error", 0.0)  # predicted - actual
                within_band = result.get("within_band", True)

                horizon_errors[delta_hours].append(float(signed_error))
                horizon_misses[delta_hours].append(not within_band)

        # Compute EWMA bias for each horizon
        alpha = 0.3  # EWMA decay: recent errors weighted more
        new_bias: dict[int, float] = {}
        new_miss: dict[int, float] = {}
        new_count: dict[int, int] = {}

        for h in range(1, PREDICTION_HOURS + 1):
            errors = horizon_errors.get(h, [])
            misses = horizon_misses.get(h, [])

            if not errors:
                continue

            new_count[h] = len(errors)

            # EWMA: more recent errors get higher weight
            ewma = errors[0]
            for e in errors[1:]:
                ewma = alpha * e + (1.0 - alpha) * ewma
            new_bias[h] = ewma

            # Band miss rate (simple average)
            new_miss[h] = sum(1 for m in misses if m) / len(misses) if misses else 0.0

        self._bias = new_bias
        self._band_miss = new_miss
        self._sample_count = new_count
        self._save_cache()

        if new_bias:
            avg_bias = np.mean(list(new_bias.values()))
            avg_miss = np.mean(list(new_miss.values())) * 100
            total = sum(new_count.values())
            logger.info(
                f"Residual learner updated: {len(new_bias)} horizons, "
                f"{total} total samples, avg bias ₹{avg_bias:+,.0f}, "
                f"band miss {avg_miss:.0f}%"
            )

    # ── Apply corrections ────────────────────────────────────────────

    def correct_predictions(self, xgb_preds: list[dict]) -> list[dict]:
        """
        Apply learned residual corrections to XGBoost predictions.

        For each hourly prediction:
          1. Subtract the EWMA bias (scaled by confidence)
          2. Widen bands if the historical miss rate is high for that horizon

        Returns a new list of corrected prediction dicts.
        """
        if not self._bias:
            return xgb_preds  # no learned data yet

        corrected = []
        for i, pred in enumerate(xgb_preds):
            horizon = i + 1
            price = pred.get("xgb_price", 0.0)
            low = pred.get("xgb_low", 0.0)
            high = pred.get("xgb_high", 0.0)

            # ── Stage 1: Bias correction ──
            bias = self._bias.get(horizon, 0.0)
            n_samples = self._sample_count.get(horizon, 0)

            if n_samples >= _MIN_SAMPLES_BIAS and price > 0:
                # Ramp correction strength from 0.3 to 0.8 based on sample count
                strength = min(
                    0.3 + 0.5 * (n_samples - _MIN_SAMPLES_BIAS)
                    / max(_MIN_SAMPLES_STRONG - _MIN_SAMPLES_BIAS, 1),
                    0.8,
                )

                # Cap correction at _MAX_CORRECTION_PCT of price
                max_corr = price * (_MAX_CORRECTION_PCT / 100.0)
                clamped_bias = max(-max_corr, min(bias, max_corr))

                correction = clamped_bias * strength
                price -= correction
                low -= correction
                high -= correction

            # ── Stage 2: Band widening for high miss-rate horizons ──
            miss_rate = self._band_miss.get(horizon, 0.0)
            if miss_rate > 0.5 and n_samples >= _MIN_SAMPLES_BIAS:
                # If >50% of predictions missed the band, widen it
                band_half = (high - low) / 2.0
                widen_factor = 1.0 + 0.5 * (miss_rate - 0.5)  # up to 25% wider
                new_half = band_half * widen_factor
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

    # ── Diagnostics ──────────────────────────────────────────────────

    def get_correction_summary(self) -> Optional[dict]:
        """Return a summary of learned corrections for display/logging."""
        if not self._bias:
            return None

        return {
            "horizons_learned": len(self._bias),
            "total_samples": sum(self._sample_count.values()),
            "avg_bias_inr": round(float(np.mean(list(self._bias.values()))), 2),
            "max_bias_inr": round(float(max(self._bias.values(), key=abs)), 2),
            "avg_band_miss_pct": round(
                float(np.mean(list(self._band_miss.values()))) * 100, 1
            ),
            "per_horizon": {
                h: {
                    "bias": round(self._bias.get(h, 0), 2),
                    "miss_rate": round(self._band_miss.get(h, 0) * 100, 1),
                    "samples": self._sample_count.get(h, 0),
                }
                for h in sorted(self._bias.keys())
            },
        }
