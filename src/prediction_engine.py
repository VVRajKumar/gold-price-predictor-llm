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

from src.config import CACHE_DIR, REFRESH_INTERVAL_MINUTES, PREDICTION_DAYS
from src.orchestrator import Orchestrator, PredictionPlan
from src.data_fetchers.market_data import MarketDataFetcher


class PredictionEngine:
    """Manages the prediction lifecycle: generate, cache, schedule, serve."""

    def __init__(self):
        self._orchestrator = Orchestrator()
        self._market = MarketDataFetcher()
        self._current_plan: Optional[PredictionPlan] = None
        self._plan_history: list[PredictionPlan] = []
        self._lock = threading.Lock()
        self._running = False

        # Try to load the latest cached plan
        self._load_cached_plan()

    # ── Cache helpers ────────────────────────────────────────────────

    @property
    def _cache_path(self) -> Path:
        return CACHE_DIR / "latest_prediction.json"

    @property
    def _history_path(self) -> Path:
        return CACHE_DIR / "prediction_history.json"

    def _save_plan(self, plan: PredictionPlan):
        self._cache_path.write_text(plan.model_dump_json(indent=2), encoding="utf-8")

        # Append to history (keep last 100)
        self._plan_history.append(plan)
        if len(self._plan_history) > 100:
            self._plan_history = self._plan_history[-100:]

        history_data = [json.loads(p.model_dump_json()) for p in self._plan_history]
        self._history_path.write_text(json.dumps(history_data, indent=2), encoding="utf-8")

    def _load_cached_plan(self):
        if self._cache_path.exists():
            try:
                data = json.loads(self._cache_path.read_text(encoding="utf-8"))
                self._current_plan = PredictionPlan(**data)
                logger.info("Loaded cached prediction plan")
            except Exception as e:
                logger.warning(f"Could not load cached plan: {e}")

        if self._history_path.exists():
            try:
                history = json.loads(self._history_path.read_text(encoding="utf-8"))
                self._plan_history = [PredictionPlan(**h) for h in history]
            except Exception:
                pass

    # ── Statistical baseline (Prophet) ──────────────────────────────

    def _prophet_baseline(self) -> list[dict]:
        """Generate a simple Prophet forecast as a sanity-check baseline."""
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

            future = model.make_future_dataframe(periods=PREDICTION_DAYS)
            forecast = model.predict(future)
            preds = forecast.tail(PREDICTION_DAYS)

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
                    "predictions": baseline,
                }

            self._current_plan = plan
            self._save_plan(plan)
            return plan

    def get_current_plan(self) -> Optional[PredictionPlan]:
        return self._current_plan

    def get_plan_history(self) -> list[PredictionPlan]:
        return self._plan_history

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
