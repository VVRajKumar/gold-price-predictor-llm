"""Prediction Engine – wraps the ML-first Orchestrator with scheduling, caching, and accuracy tracking."""

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
)
from .orchestrator import Orchestrator, PredictionPlan
from .data_fetchers.market_data import MarketDataFetcher
from .accuracy_tracker import AccuracyTracker
from . import cloud_storage
from .time_utils import iso_now_ist


class PredictionEngine:
    """Manages the prediction lifecycle: generate, cache, schedule, serve."""

    def __init__(self):
        # Restore persisted accuracy data from cloud before anything reads cache
        cloud_storage.sync_from_cloud()

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
        return CACHE_DIR / "weekly_workflow_reset_v2.marker"

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
        content = json.loads(plan.model_dump_json())
        cloud_storage.persist("latest_prediction.json", content)

    def _load_cached_plan(self):
        if self._cache_path.exists():
            try:
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                plan = PredictionPlan(**data)
                if not np.isfinite(plan.current_price) or plan.current_price <= 0:
                    logger.warning("Ignoring cached prediction plan with invalid current_price")
                elif plan.current_price < 30_000:
                    # Price is clearly not in INR/10g scale (likely stale USD-era cache)
                    logger.warning(
                        f"Cached plan current_price ₹{plan.current_price:,.2f} is below "
                        f"₹30,000 – likely stale USD-era cache; discarding"
                    )
                else:
                    # Re-apply guardrails so confidence decay, band widening,
                    # and XGBoost noise are enforced on old stored plans too.
                    from .guardrails import validate_prediction_plan
                    plan_dict = json.loads(plan.model_dump_json())
                    corrected = validate_prediction_plan(
                        plan_dict, plan.current_price, len(plan.daily_predictions)
                    )
                    plan = PredictionPlan(**corrected)
                    self._current_plan = plan
                    logger.info("Loaded cached prediction plan (guardrails re-applied)")
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

    # ── Core methods ────────────────────────────────────────────────

    def generate(self) -> PredictionPlan:
        """Run the full ML-first prediction pipeline (blocks until done)."""
        with self._lock:
            logger.info("PredictionEngine: generating new ML-first prediction …")
            plan = self._orchestrator.generate_prediction()

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
                prev_plan = self._current_plan  # keep as fallback
                try:
                    plan = self._orchestrator.generate_prediction()

                    # Validate the new plan before accepting it
                    if (not np.isfinite(plan.current_price)
                            or plan.current_price <= 0
                            or plan.current_price < 30_000):
                        raise ValueError(
                            f"Generated plan has invalid current_price: {plan.current_price}"
                        )

                    # ML ensemble now handles all price prediction internally.
                    # No separate XGBoost baseline or blending needed.

                    self._current_plan = plan
                    self._save_plan(plan)

                    try:
                        tracker = self.get_accuracy_tracker()
                        tracker.store_plan(json.loads(plan.model_dump_json()))
                        # Update residual learner with latest accuracy data
                        ml_ensemble = self._orchestrator._ml_ensemble
                        ml_ensemble.update_residuals(tracker.get_all_evaluations())
                    except Exception as e:
                        logger.warning(f"Could not store hourly plan for accuracy: {e}")

                except Exception as e:
                    logger.error(f"Prediction generation failed: {e}")
                    if prev_plan is not None:
                        logger.info("Keeping previous valid plan as fallback")
                        self._current_plan = prev_plan

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
            self._accuracy.start_auto_check(interval_hours=1)
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
