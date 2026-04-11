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
from .time_utils import (
    iso_now_ist, now_ist,
    current_slot_ist, next_slot_ist, slot_id,
    is_market_open, next_market_open_ist,
)


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
        # Strip large fields that aren't needed for plan persistence/display
        # and bloat the Gist payload.
        agent_reports = content.get("agent_reports") or {}
        # Remove hourly SHAP drivers (large per-hour breakdown; feature_importance kept)
        ml_report = agent_reports.get("_ml_ensemble") or {}
        shap_data = ml_report.get("shap") or {}
        shap_data.pop("hourly_drivers", None)
        # Remove verbose data_points from every agent report
        for report in agent_reports.values():
            if isinstance(report, dict):
                report.pop("data_points", None)
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
                    saved_reports = plan_dict.get("agent_reports", {})
                    corrected = validate_prediction_plan(
                        plan_dict, plan.current_price, len(plan.daily_predictions)
                    )
                    plan = PredictionPlan(**corrected)
                    # Restore agent_reports (guardrails don't handle these)
                    if saved_reports:
                        plan.agent_reports = saved_reports
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
            if not is_market_open():
                logger.info("Market closed (weekend) — skipping generation")
                if self._current_plan is not None:
                    return self._current_plan
                raise RuntimeError(
                    "Cannot generate prediction: market is closed and no cached plan exists. "
                    "A prediction will be generated automatically when the market reopens on Monday."
                )
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
        """Return an active plan, regenerating only when the 6-hour slot changes.

        Predictions are aligned to fixed IST slots (00:00, 06:00, 12:00, 18:00).
        A new prediction is only generated when:
          - No plan exists at all
          - The current plan belongs to a different 6-hour slot
        On weekends (Saturday/Sunday IST) the gold market is closed, so no new
        prediction is generated — the most recent weekday plan is returned instead.
        """
        with self._lock:
            # Skip automatic generation on weekends (market closed)
            if not is_market_open():
                if self._current_plan is not None:
                    logger.info(
                        "Weekend (market closed) — returning last weekday prediction"
                    )
                    return self._current_plan
                logger.info(
                    "Weekend (market closed) and no cached plan — skipping generation"
                )
                return None

            needs_generation = self._current_plan is None
            if self._current_plan is not None:
                try:
                    gen_dt = datetime.fromisoformat(self._current_plan.generated_at)
                    plan_slot = slot_id(gen_dt)
                    current_slot = slot_id(now_ist())
                    needs_generation = plan_slot != current_slot
                    if needs_generation:
                        logger.info(
                            f"Plan slot {plan_slot} differs from current slot {current_slot} "
                            f"— will regenerate"
                        )
                except Exception:
                    needs_generation = True

            if needs_generation:
                logger.info("Generating prediction for current 6-hour slot")
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
            self._accuracy.start_auto_check(interval_hours=2)
        return self._accuracy

    # ── Auto-refresh loop ───────────────────────────────────────────

    def start_auto_refresh(self):
        """Start a background thread that regenerates predictions on 6-hour slot boundaries.

        Predictions are aligned to 00:00, 06:00, 12:00, 18:00 IST.
        On weekends (Saturday/Sunday IST) the thread sleeps until Monday
        since the gold market is closed and prices do not change.
        """
        if self._running:
            return
        self._running = True

        def _loop():
            while self._running:
                try:
                    now = now_ist()

                    # If it's a weekend / Monday pre-market, sleep until Monday 09:00 IST
                    if not is_market_open(now):
                        next_open = next_market_open_ist(now)
                        wait_seconds = max(60, (next_open - now).total_seconds() + 30)
                        logger.info(
                            f"Auto-refresh: market closed — sleeping until "
                            f"{next_open.strftime('%H:%M %b %d')} IST "
                            f"({wait_seconds / 3600:.1f} hours)"
                        )
                        time.sleep(wait_seconds)
                        if not self._running:
                            break
                        # Woke at Monday market open — generate immediately
                        logger.info("Auto-refresh: market reopened — generating prediction")
                        self.generate()
                        continue

                    # Sleep until the next 6-hour slot boundary
                    next_slot = next_slot_ist()
                    wait_seconds = max(60, (next_slot - now).total_seconds() + 30)
                    logger.info(
                        f"Auto-refresh: next prediction at {next_slot.strftime('%H:%M %b %d')} IST "
                        f"(sleeping {wait_seconds / 60:.0f} min)"
                    )
                    time.sleep(wait_seconds)
                    if not self._running:
                        break
                    # Re-check after sleeping: the next slot may have landed
                    # on a weekend (e.g. Friday 18:00 → Saturday 00:00).
                    if not is_market_open():
                        logger.info(
                            "Auto-refresh: woke on weekend/holiday — skipping generation"
                        )
                        continue
                    self.generate()
                except Exception as e:
                    logger.error(f"Auto-refresh failed: {e}")
                    time.sleep(300)  # Back off 5 minutes on error

        t = threading.Thread(target=_loop, daemon=True, name="prediction-refresh")
        t.start()
        logger.info("Auto-refresh started — aligned to 6-hour IST slots (00:00, 06:00, 12:00, 18:00), weekends skipped")

    def stop_auto_refresh(self):
        self._running = False
