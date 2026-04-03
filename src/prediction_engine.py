"""
Prediction Engine – wraps the Orchestrator with scheduling, caching,
and a Prophet-based statistical baseline for sanity-checking LLM forecasts.
"""

from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd
import numpy as np
from loguru import logger

from .config import (
    CACHE_DIR,
    REFRESH_INTERVAL_MINUTES,
    PREDICTION_HOURS,
    PLAN_REFRESH_HOURS,
    FORECAST_GRANULARITY,
)
from .orchestrator import Orchestrator, PredictionPlan
from .data_fetchers.market_data import MarketDataFetcher
from .accuracy_tracker import AccuracyTracker
from .time_utils import iso_now_ist


class PredictionEngine:
    """Manages the prediction lifecycle: generate, cache, schedule, serve."""

    def __init__(self):
        self._orchestrator = Orchestrator()
        self._market = MarketDataFetcher()
        self._accuracy: Optional[AccuracyTracker] = None
        self._current_plan: Optional[PredictionPlan] = None
        self._weekly_archive: list[dict] = []
        self._lock = threading.Lock()
        self._running = False

        # One-time storage reset requested for the new weekly workflow.
        self._reset_storage_once()

        # Try to load the latest cached plan and weekly archive.
        self._load_cached_plan()
        self._load_weekly_archive()

    # ── Cache helpers ────────────────────────────────────────────────

    @property
    def _cache_path(self) -> Path:
        return CACHE_DIR / "latest_prediction.json"

    @property
    def _history_path(self) -> Path:
        return CACHE_DIR / "prediction_history.json"

    @property
    def _weekly_archive_path(self) -> Path:
        return CACHE_DIR / "weekly_prediction_archive.json"

    @property
    def _reset_marker_path(self) -> Path:
        return CACHE_DIR / "weekly_workflow_reset_v1.marker"

    def _week_id(self, ts: str) -> str:
        dt = datetime.fromisoformat(ts)
        iso = dt.isocalendar()
        return f"{iso.year}-W{iso.week:02d}"

    def _reset_storage_once(self):
        if self._reset_marker_path.exists():
            return

        for p in [
            CACHE_DIR / "latest_prediction.json",
            CACHE_DIR / "prediction_history.json",
            CACHE_DIR / "accuracy_log.json",
            CACHE_DIR / "stored_plans.json",
            CACHE_DIR / "weekly_prediction_archive.json",
        ]:
            try:
                if p.exists():
                    p.unlink()
            except Exception as e:
                logger.warning(f"Could not reset file {p.name}: {e}")

        self._reset_marker_path.write_text(iso_now_ist(), encoding="utf-8")
        logger.info("Storage reset complete for weekly workflow")

    def _load_weekly_archive(self):
        if self._weekly_archive_path.exists():
            try:
                data = json.loads(self._weekly_archive_path.read_text(encoding="utf-8"))
                if isinstance(data, list):
                    self._weekly_archive = data
            except Exception as e:
                logger.warning(f"Could not load weekly archive: {e}")

    def _save_weekly_archive(self):
        self._weekly_archive_path.write_text(
            json.dumps(self._weekly_archive, indent=2),
            encoding="utf-8",
        )

    def _save_plan(self, plan: PredictionPlan):
        self._cache_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

    def _load_cached_plan(self):
        if self._cache_path.exists():
            try:
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                plan = PredictionPlan(**data)
                if not np.isfinite(plan.current_price) or plan.current_price <= 0:
                    logger.warning("Ignoring cached prediction plan with invalid current_price")
                else:
                    self._current_plan = plan
                    logger.info("Loaded cached prediction plan")
            except Exception as e:
                logger.warning(f"Could not load cached plan: {e}")

    def _archive_previous_week_if_needed(self):
        if self._current_plan is None:
            return

        try:
            current_week = self._week_id(iso_now_ist())
            plan_week = self._week_id(self._current_plan.generated_at)
            if plan_week == current_week:
                return

            if not any(item.get("week_id") == plan_week for item in self._weekly_archive):
                self._weekly_archive.append({
                    "week_id": plan_week,
                    "archived_at": iso_now_ist(),
                    "plan": json.loads(self._current_plan.model_dump_json()),
                })
                self._weekly_archive = self._weekly_archive[-52:]
                self._save_weekly_archive()
                logger.info(f"Archived completed weekly prediction: {plan_week}")

            # Do not keep old-week plan as current.
            self._current_plan = None
            try:
                if self._cache_path.exists():
                    self._cache_path.unlink()
            except Exception:
                pass
        except Exception as e:
            logger.warning(f"Weekly archive check failed: {e}")

    # ── Statistical baseline (Prophet) ──────────────────────────────

    def _prophet_baseline(self) -> list[dict]:
        """Generate a simple Prophet forecast as a sanity-check baseline."""
        if FORECAST_GRANULARITY.lower() == "hourly":
            # Daily Prophet baseline is not aligned with hourly horizon.
            return []

        try:
            from prophet import Prophet

            df = self._market.fetch_ticker("GC=F", period_days=365)
            if df.empty or len(df) < 60:
                return []

            pdf = pd.DataFrame({"ds": df.index, "y": df["Close"].values})
            pdf = pdf.dropna()

            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,
            )
            model.fit(pdf)

            future = model.make_future_dataframe(periods=PREDICTION_HOURS)
            forecast = model.predict(future)
            preds = forecast.tail(PREDICTION_HOURS)

            return [
                {
                    "date": row["ds"].strftime("%Y-%m-%d"),
                    "prophet_price": round(row["yhat"], 2),
                    "prophet_low": round(row["yhat_lower"], 2),
                    "prophet_high": round(row["yhat_upper"], 2),
                }
                for _, row in preds.iterrows()
            ]
        except ImportError:
            logger.warning("Prophet not installed – skipping baseline")
            return []
        except Exception as e:
            logger.warning(f"Prophet baseline failed: {e}")
            return []

    # ── Core methods ────────────────────────────────────────────────

    def generate(self) -> PredictionPlan:
        """Run the full prediction pipeline (blocks until done)."""
        with self._lock:
            logger.info("PredictionEngine: generating new prediction …")
            plan = self._orchestrator.generate_prediction()

            # Attach Prophet baseline as extra data
            baseline = self._prophet_baseline()
            if baseline:
                plan.agent_reports["prophet_baseline"] = {
                    "type": "statistical_model",
                    "summary": "Prophet baseline generated from recent gold price time-series trend.",
                    "predictions": baseline,
                }

            self._current_plan = plan
            self._save_plan(plan)

            # Store plan for accuracy tracking
            try:
                tracker = self.get_accuracy_tracker()
                tracker.store_plan(json.loads(plan.model_dump_json()))
            except Exception as e:
                logger.warning(f"Could not store plan for accuracy: {e}")

            return plan

    def ensure_hourly_prediction(self) -> Optional[PredictionPlan]:
        """Return an active plan, refreshing automatically when it gets stale by hour."""
        with self._lock:
            stale = self._current_plan is None
            if self._current_plan is not None and not stale:
                try:
                    gen_dt = datetime.fromisoformat(self._current_plan.generated_at)
                    now_dt = datetime.fromisoformat(iso_now_ist())
                    stale = (now_dt - gen_dt).total_seconds() >= PLAN_REFRESH_HOURS * 3600
                except Exception:
                    stale = True

            if stale:
                logger.info("No active hourly plan found (or stale) — generating new prediction")
                plan = self._orchestrator.generate_prediction()

                baseline = self._prophet_baseline()
                if baseline:
                    plan.agent_reports["prophet_baseline"] = {
                        "type": "statistical_model",
                        "summary": "Prophet baseline generated from recent gold price time-series trend.",
                        "predictions": baseline,
                    }

                self._current_plan = plan
                self._save_plan(plan)

                try:
                    tracker = self.get_accuracy_tracker()
                    tracker.store_plan(json.loads(plan.model_dump_json()))
                except Exception as e:
                    logger.warning(f"Could not store hourly plan for accuracy: {e}")

            return self._current_plan

    def ensure_weekly_prediction(self) -> Optional[PredictionPlan]:
        """Backward-compatible wrapper retained for existing callers."""
        return self.ensure_hourly_prediction()

    def get_current_plan(self) -> Optional[PredictionPlan]:
        return self._current_plan

    def get_plan_history(self) -> list[PredictionPlan]:
        # Main page no longer shows rolling prediction history.
        return []

    def get_weekly_archive(self) -> list[dict]:
        return list(reversed(self._weekly_archive))

    def get_accuracy_tracker(self) -> AccuracyTracker:
        if self._accuracy is None:
            self._accuracy = AccuracyTracker()
            self._accuracy.start_auto_check(interval_hours=6)
        return self._accuracy

    # ── Auto-refresh loop ───────────────────────────────────────────

    def start_auto_refresh(self):
        """Start a background thread that regenerates predictions on a schedule."""
        if self._running:
            return
        self._running = True

        def _loop():
            while self._running:
                try:
                    self.generate()
                except Exception as e:
                    logger.error(f"Auto-refresh failed: {e}")
                time.sleep(REFRESH_INTERVAL_MINUTES * 60)

        t = threading.Thread(target=_loop, daemon=True, name="prediction-refresh")
        t.start()
        logger.info(
            f"Auto-refresh started – regenerating every {REFRESH_INTERVAL_MINUTES} min"
        )

    def stop_auto_refresh(self):
        self._running = False
