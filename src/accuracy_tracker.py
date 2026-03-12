"""
Accuracy Tracker – compares past predictions against actual gold prices
and computes accuracy metrics (MAE, MAPE, directional accuracy, hit-rate).
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

from src.config import CACHE_DIR
from src.data_fetchers.market_data import MarketDataFetcher


_ACCURACY_PATH = CACHE_DIR / "accuracy_log.json"


class AccuracyTracker:
    """Track and score past predictions vs actual prices."""

    def __init__(self):
        self._market = MarketDataFetcher()
        self._log: list[dict] = self._load_log()

    # ── Persistence ──────────────────────────────────────────────────

    def _load_log(self) -> list[dict]:
        if _ACCURACY_PATH.exists():
            try:
                return json.loads(_ACCURACY_PATH.read_text(encoding="utf-8"))
            except Exception:
                return []
        return []

    def _save_log(self):
        _ACCURACY_PATH.write_text(
            json.dumps(self._log, indent=2, default=str), encoding="utf-8"
        )

    # ── Core: evaluate a prediction plan ─────────────────────────────

    def evaluate_plan(self, plan_dict: dict) -> Optional[dict]:
        """
        Given a serialised PredictionPlan, look up actual prices for each
        predicted day that is already in the past and compute metrics.
        Returns a results dict or None if no days are evaluable yet.
        """
        preds = plan_dict.get("daily_predictions", [])
        if not preds:
            return None

        generated_at = plan_dict.get("generated_at", "")
        current_price = plan_dict.get("current_price", 0)

        # Fetch recent actual gold prices
        gold_df = self._market.fetch_ticker("GC=F", period_days=30)
        if gold_df.empty:
            return None

        close = gold_df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        today = datetime.now().date()
        evaluated_days = []

        for dp in preds:
            pred_date = pd.to_datetime(dp["date"]).date()
            if pred_date >= today:
                continue  # future day, can't evaluate yet

            # Find actual close for that date (or nearest prior trading day)
            actual = self._get_actual_price(close, pred_date)
            if actual is None:
                continue

            predicted = dp["predicted_price"]
            low = dp["low_range"]
            high = dp["high_range"]
            error = predicted - actual
            abs_error = abs(error)
            pct_error = (abs_error / actual) * 100 if actual != 0 else 0
            within_band = low <= actual <= high

            evaluated_days.append({
                "date": str(pred_date),
                "predicted": round(predicted, 2),
                "actual": round(actual, 2),
                "low_range": round(low, 2),
                "high_range": round(high, 2),
                "error": round(error, 2),
                "abs_error": round(abs_error, 2),
                "pct_error": round(pct_error, 2),
                "within_band": within_band,
            })

        if not evaluated_days:
            return None

        # Compute aggregate metrics
        errors = [d["abs_error"] for d in evaluated_days]
        pct_errors = [d["pct_error"] for d in evaluated_days]
        band_hits = [d["within_band"] for d in evaluated_days]

        # Directional accuracy: did we predict the right direction from current_price?
        directional_correct = 0
        directional_total = 0
        for d in evaluated_days:
            pred_dir = "up" if d["predicted"] >= current_price else "down"
            actual_dir = "up" if d["actual"] >= current_price else "down"
            directional_total += 1
            if pred_dir == actual_dir:
                directional_correct += 1

        result = {
            "plan_generated_at": generated_at,
            "evaluated_at": datetime.now().isoformat(),
            "current_price_at_prediction": round(current_price, 2),
            "days_evaluated": len(evaluated_days),
            "days_total": len(preds),
            "mae": round(np.mean(errors), 2),
            "mape": round(np.mean(pct_errors), 2),
            "max_error": round(max(errors), 2),
            "band_hit_rate": round(sum(band_hits) / len(band_hits) * 100, 1),
            "directional_accuracy": round(
                directional_correct / directional_total * 100, 1
            ) if directional_total > 0 else 0,
            "daily_results": evaluated_days,
        }

        # Save to log (avoid duplicates for same plan)
        existing_ids = {r["plan_generated_at"] for r in self._log}
        if generated_at not in existing_ids:
            self._log.append(result)
        else:
            # Update the existing entry
            for i, r in enumerate(self._log):
                if r["plan_generated_at"] == generated_at:
                    self._log[i] = result
                    break
        self._save_log()
        return result

    def _get_actual_price(self, close_series: pd.Series, target_date) -> Optional[float]:
        """Get the closing price for a date, checking +/- 2 days for weekends."""
        for offset in range(0, 4):
            check = target_date - timedelta(days=offset)
            idx = pd.Timestamp(check)
            if idx in close_series.index:
                val = close_series.loc[idx]
                return float(val)
        return None

    # ── Retrieve all past evaluations ────────────────────────────────

    def get_all_evaluations(self) -> list[dict]:
        return self._log

    def get_latest_evaluation(self) -> Optional[dict]:
        return self._log[-1] if self._log else None

    def get_aggregate_stats(self) -> Optional[dict]:
        """Compute aggregate accuracy across ALL historical evaluations."""
        if not self._log:
            return None

        all_days = []
        for ev in self._log:
            all_days.extend(ev.get("daily_results", []))

        if not all_days:
            return None

        errors = [d["abs_error"] for d in all_days]
        pct_errors = [d["pct_error"] for d in all_days]
        band_hits = [d["within_band"] for d in all_days]

        return {
            "total_predictions_evaluated": len(all_days),
            "total_plans_evaluated": len(self._log),
            "overall_mae": round(np.mean(errors), 2),
            "overall_mape": round(np.mean(pct_errors), 2),
            "overall_band_hit_rate": round(sum(band_hits) / len(band_hits) * 100, 1),
            "best_mae": round(min(r["mae"] for r in self._log), 2),
            "worst_mae": round(max(r["mae"] for r in self._log), 2),
            "avg_directional_accuracy": round(
                np.mean([r["directional_accuracy"] for r in self._log]), 1
            ),
        }
