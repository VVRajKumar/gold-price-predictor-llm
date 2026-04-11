"""
Accuracy Tracker – compares past predictions against actual gold prices
and computes accuracy metrics (MAE, MAPE, directional accuracy, hit-rate).
Auto-refreshes when market closes each day.
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

from .config import CACHE_DIR
from .data_fetchers.market_data import MarketDataFetcher
from .time_utils import iso_now_ist, now_ist, IST_OFFSET
from . import cloud_storage


_ACCURACY_PATH = CACHE_DIR / "accuracy_log.json"
_PLANS_STORE_PATH = CACHE_DIR / "stored_plans.json"
_ARCHIVE_PATH = CACHE_DIR / "prediction_archive.json"

# Only evaluate/store plans generated on or after this cutoff (hourly scorecard start)
_HOURLY_SCORECARD_START = datetime(2026, 4, 4, 0, 0, 0)

# Directional neutral zone: if actual moved less than this %, the market
# is considered flat and the prediction is counted as directionally correct.
_DIR_NEUTRAL_PCT = 0.01

# Bump this version string whenever guardrail parameters change.
# Triggers a one-time rebase that retroactively widens bands on all stored
# plans so accuracy metrics reflect the new parameters immediately.
_GUARDRAIL_VERSION = "v5_ist_timezone_fix_20260411"


class AccuracyTracker:
    """Track and score past predictions vs actual prices."""

    def __init__(self):
        self._market = MarketDataFetcher()
        self._log: list[dict] = self._load_log()
        self._stored_plans: list[dict] = self._load_stored_plans()
        self._archive: list[dict] = self._load_archive()
        self._last_checked: Optional[str] = None
        self._auto_running = False
        self._purge_stale_entries()
        self._rebase_guardrails()
        self._backfill_archive_from_log()

    # ── Purge pre-INR / pre-cutoff entries ───────────────────────────

    def _purge_stale_entries(self):
        """Remove accuracy log entries from before the INR migration cutoff
        or with predictions in USD scale (< ₹30,000)."""
        before = len(self._log)
        self._log = [
            e for e in self._log
            if self._is_valid_entry(e)
        ]
        if len(self._log) < before:
            logger.info(f"Purged {before - len(self._log)} stale accuracy entries")
            self._save_log()

    def _is_valid_entry(self, entry: dict) -> bool:
        gen = entry.get("plan_generated_at", "")
        try:
            gen_dt = datetime.fromisoformat(str(gen))
            if gen_dt.replace(tzinfo=None) < _HOURLY_SCORECARD_START:
                return False
        except (ValueError, TypeError):
            return False
        # Also reject entries with USD-scale predictions
        for d in entry.get("daily_results", []):
            if d.get("predicted", 0) < 30000:
                return False
        return True

    # ── Retroactive guardrail rebase ─────────────────────────────────

    def _rebase_guardrails(self):
        """One-time rebase: adjust bands on stored plans when guardrails change.

        Retroactively applies the current horizon-aware band envelope so that
        accuracy metrics (especially band_hit_rate) reflect the updated
        prediction parameters immediately.

        Uses a version marker file so the rebase only runs once per
        guardrail parameter change (bump ``_GUARDRAIL_VERSION`` to re-run).
        """
        marker_path = CACHE_DIR / "guardrail_rebase.marker"
        try:
            if (marker_path.exists()
                    and marker_path.read_text(encoding="utf-8").strip() == _GUARDRAIL_VERSION):
                return  # already rebased for this version
        except Exception:
            pass

        if not self._stored_plans:
            # Write marker even when there are no plans so we don't
            # retry on every startup.
            try:
                marker_path.write_text(_GUARDRAIL_VERSION, encoding="utf-8")
            except Exception:
                pass
            return

        from .guardrails import validate_prediction_plan, _band_envelope_pct

        bands_widened = 0
        for plan in self._stored_plans:
            current_price = plan.get("current_price", 0)
            preds = plan.get("daily_predictions", [])
            if not preds or current_price <= 0:
                continue

            # Step 1: standard guardrails (enforces new min-band 1%, wider caps)
            validate_prediction_plan(plan, current_price, len(preds))

            # Step 2: horizon-aware band widening using the new envelope.
            # _band_envelope_pct(h) returns the max-allowed band half-width
            # as a fraction (e.g. 0.02 = 2%).  We use it as a *minimum*
            # here so bands that were crushed by old double-clamping get
            # expanded to the intended width.
            for h, dp in enumerate(preds):
                predicted = dp.get("predicted_price", current_price)
                low = dp.get("low_range", predicted)
                high = dp.get("high_range", predicted)

                if predicted <= 0:
                    continue

                min_half = predicted * _band_envelope_pct(h)
                current_half = abs(high - low) / 2.0

                if current_half < min_half:
                    dp["low_range"] = round(predicted - min_half, 2)
                    dp["high_range"] = round(predicted + min_half, 2)
                    bands_widened += 1

        self._save_stored_plans()

        # Clear accuracy log so all plans are re-evaluated with the wider
        # bands on the next refresh_all() call.
        self._log = []
        self._save_log()

        try:
            marker_path.write_text(_GUARDRAIL_VERSION, encoding="utf-8")
        except Exception:
            pass

        logger.info(
            f"Guardrail rebase ({_GUARDRAIL_VERSION}): widened {bands_widened} "
            f"prediction bands across {len(self._stored_plans)} plans, "
            f"cleared accuracy log for re-evaluation"
        )

    # ── Backfill archive from accuracy log ───────────────────────────

    def _backfill_archive_from_log(self):
        """Ensure all evaluated hourly results from the accuracy log are
        also present in the prediction archive.

        The archive was introduced after the accuracy log, so some older
        evaluations (e.g. April 4-6) may only exist in accuracy_log.json.
        This method copies any missing entries into the permanent archive.
        """
        if not self._log:
            return

        # Build set of existing archive keys for fast lookup.
        # Key = (plan_generated_at, date) uniquely identifies an hourly
        # prediction from a specific plan generation run.
        existing_keys: set[tuple[str, str]] = set()
        for entry in self._archive:
            existing_keys.add(
                (entry.get("plan_generated_at", ""), entry.get("date", ""))
            )

        added = 0
        for evaluation in self._log:
            gen_at = evaluation.get("plan_generated_at", "")
            for d in evaluation.get("daily_results", []):
                key = (gen_at, d.get("date", ""))
                if key not in existing_keys:
                    entry_data = {
                        "plan_generated_at": gen_at,
                        "evaluated_at": evaluation.get("evaluated_at", ""),
                        "current_price_at_prediction": evaluation.get(
                            "current_price_at_prediction", 0
                        ),
                        **d,
                    }
                    self._archive.append(entry_data)
                    existing_keys.add(key)
                    added += 1

        if added:
            self._save_archive()
            logger.info(
                f"Archive backfill: added {added} entries from accuracy log "
                f"(total archive: {len(self._archive)})"
            )

    # ── Persistence ──────────────────────────────────────────────────

    def _load_log(self) -> list[dict]:
        if _ACCURACY_PATH.exists():
            try:
                return json.loads(_ACCURACY_PATH.read_text(encoding="utf-8"))
            except Exception:
                return []
        # Fallback: try restoring from cloud
        data = cloud_storage.load("accuracy_log.json")
        return data if isinstance(data, list) else []

    def _save_log(self):
        cloud_storage.persist("accuracy_log.json", self._log)

    def _load_stored_plans(self) -> list[dict]:
        if _PLANS_STORE_PATH.exists():
            try:
                return json.loads(_PLANS_STORE_PATH.read_text(encoding="utf-8"))
            except Exception:
                return []
        # Fallback: try restoring from cloud
        data = cloud_storage.load("stored_plans.json")
        return data if isinstance(data, list) else []

    def _save_stored_plans(self):
        cloud_storage.persist("stored_plans.json", self._stored_plans)

    # ── Prediction archive (unlimited historical log) ────────────────

    def _load_archive(self) -> list[dict]:
        if _ARCHIVE_PATH.exists():
            try:
                return json.loads(_ARCHIVE_PATH.read_text(encoding="utf-8"))
            except Exception:
                return []
        data = cloud_storage.load("prediction_archive.json")
        return data if isinstance(data, list) else []

    def _save_archive(self):
        cloud_storage.persist("prediction_archive.json", self._archive)

    def _archive_evaluation(self, result: dict):
        """Append evaluated hourly results to the permanent archive.

        Unlike the rolling accuracy log (capped at _MAX_LOG_ENTRIES), the
        archive keeps **all** historical evaluated hours forever so the
        Prediction Archive page can display long-term performance.
        Deduplication is by (plan_generated_at, date) pair.
        """
        gen_at = result.get("plan_generated_at", "")
        existing_keys: set[tuple[str, str]] = set()
        for entry in self._archive:
            existing_keys.add((entry.get("plan_generated_at", ""), entry.get("date", "")))

        added = 0
        updated = 0
        for d in result.get("daily_results", []):
            key = (gen_at, d.get("date", ""))
            entry_data = {
                "plan_generated_at": gen_at,
                "evaluated_at": result.get("evaluated_at", ""),
                "current_price_at_prediction": result.get("current_price_at_prediction", 0),
                **d,
            }
            if key not in existing_keys:
                self._archive.append(entry_data)
                existing_keys.add(key)
                added += 1
            else:
                # Update existing entry with latest evaluation
                for i, entry in enumerate(self._archive):
                    if (entry.get("plan_generated_at", ""), entry.get("date", "")) == key:
                        self._archive[i] = entry_data
                        updated += 1
                        break

        if added or updated:
            self._save_archive()
            logger.debug(f"Archive: {added} new, {updated} updated (total: {len(self._archive)})")

    def get_prediction_archive(self) -> list[dict]:
        """Return the full prediction archive for the Archive page."""
        return self._archive

    # ── Store a plan for future evaluation ───────────────────────────

    # Fields to keep in stored plans (everything else is stripped to reduce payload)
    _PLAN_KEEP_FIELDS = {"generated_at", "current_price", "daily_predictions", "overall_outlook", "overall_confidence"}
    _MAX_STORED_PLANS = 20
    _MAX_LOG_ENTRIES = 30

    def store_plan(self, plan_dict: dict):
        """Persist a prediction plan so it can be re-evaluated later as days pass."""
        gen_at = plan_dict.get("generated_at", "")
        # Skip plans generated before the hourly scorecard cutoff
        try:
            gen_dt = datetime.fromisoformat(str(gen_at))
            if gen_dt.replace(tzinfo=None) < _HOURLY_SCORECARD_START:
                logger.debug(f"Skipping pre-cutoff plan {gen_at}")
                return
        except (ValueError, TypeError):
            pass
        existing_ids = {p.get("generated_at") for p in self._stored_plans}
        if gen_at not in existing_ids:
            # Strip bulky fields not needed for accuracy evaluation
            slim_plan = {k: v for k, v in plan_dict.items() if k in self._PLAN_KEEP_FIELDS}
            self._stored_plans.append(slim_plan)
            # Keep last _MAX_STORED_PLANS plans
            if len(self._stored_plans) > self._MAX_STORED_PLANS:
                self._stored_plans = self._stored_plans[-self._MAX_STORED_PLANS:]
            self._save_stored_plans()
            logger.info(f"Stored plan {gen_at} for future accuracy tracking")

    def get_stored_plans(self) -> list[dict]:
        return self._stored_plans

    def delete_plan_entry(self, timestamp: str) -> int:
        """Delete stored plan(s) and accuracy log entries matching a timestamp.

        Returns the number of entries removed.
        """
        removed = 0
        before_plans = len(self._stored_plans)
        self._stored_plans = [
            p for p in self._stored_plans if p.get("generated_at", "") != timestamp
        ]
        removed += before_plans - len(self._stored_plans)

        before_log = len(self._log)
        self._log = [
            e for e in self._log if e.get("plan_generated_at", "") != timestamp
        ]
        removed += before_log - len(self._log)

        if removed:
            self._save_stored_plans()
            self._save_log()
            logger.info(f"Deleted {removed} entries matching timestamp '{timestamp}'")
        return removed

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
        # Skip evaluation for plans generated before the hourly scorecard cutoff
        try:
            gen_dt = datetime.fromisoformat(str(generated_at))
            if gen_dt.replace(tzinfo=None) < _HOURLY_SCORECARD_START:
                return None
        except (ValueError, TypeError):
            pass
        current_price = plan_dict.get("current_price", 0)

        # Fetch recent actual gold prices on hourly candles and convert to INR/10g
        gold_df = self._market.fetch_ticker("GC=F", period_days=10, interval="1h")
        if gold_df.empty:
            return None

        # Use time-aligned daily FX rates for accurate conversion
        gold_df = self._market.convert_usd_to_inr(gold_df, period_days=15)

        # Convert gold data index from UTC to IST.
        # yfinance returns UTC timestamps (tz-stripped via tz_convert(None)).
        # Prediction timestamps are in IST, so we must align the index.
        # After adding the IST offset, floor to the nearest hour so each
        # UTC candle maps to a unique IST hour (max 30-min rounding error).
        if not gold_df.empty and isinstance(gold_df.index, pd.DatetimeIndex):
            gold_df.index = gold_df.index + pd.Timedelta(IST_OFFSET)
            gold_df.index = gold_df.index.floor("h")
            # Drop duplicate hours that may arise from rounding
            gold_df = gold_df[~gold_df.index.duplicated(keep="last")]

        close = gold_df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]

        now_ts = now_ist().replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)
        evaluated_days = []

        for dp in preds:
            pred_ts = pd.to_datetime(dp["date"], errors="coerce")
            if pd.isna(pred_ts):
                continue
            pred_ts = pred_ts.tz_localize(None) if getattr(pred_ts, "tzinfo", None) is not None else pred_ts
            if pred_ts >= now_ts:
                continue  # future hour, can't evaluate yet

            # Find actual close for the prediction hour (or nearest prior hour)
            actual = self._get_actual_price(close, pred_ts)
            if actual is None:
                continue

            predicted = dp["predicted_price"]
            low = dp["low_range"]
            high = dp["high_range"]

            # Guard against None / non-numeric values from cached data
            try:
                predicted = float(predicted)
                low = float(low)
                high = float(high)
            except (TypeError, ValueError):
                continue

            error = predicted - actual
            abs_error = abs(error)
            pct_error = (abs_error / actual) * 100 if actual != 0 else 0
            within_band = low <= actual <= high

            evaluated_days.append({
                "date": pred_ts.strftime("%Y-%m-%d %H:%M:%S"),
                "predicted": round(predicted, 2),
                "actual": round(actual, 2),
                "low_range": round(low, 2),
                "high_range": round(high, 2),
                "error": round(error, 2),
                "pct_error": round(pct_error, 2),
                "within_band": within_band,
                "plan_generated_at": generated_at,
            })

        if not evaluated_days:
            return None

        # Compute aggregate metrics
        errors = [abs(d["error"]) for d in evaluated_days]
        pct_errors = [d["pct_error"] for d in evaluated_days]
        band_hits = [d["within_band"] for d in evaluated_days]

        # Directional accuracy: hour-over-hour from the evaluated series.
        # Sort by time, then compare consecutive hours to check if the
        # predicted direction matches the actual direction of movement.
        # This is consistent with the aggregate scorecard computation.
        # Neutral zone: if actual moved < 0.05% either way, count it as
        # correct regardless of prediction (the market was essentially flat).
        sorted_results = sorted(evaluated_days, key=lambda d: d.get("date", ""))
        directional_correct = 0
        directional_total = 0
        for idx in range(1, len(sorted_results)):
            prev_actual = sorted_results[idx - 1].get("actual", 0)
            curr_predicted = sorted_results[idx].get("predicted", 0)
            curr_actual = sorted_results[idx].get("actual", 0)
            if prev_actual <= 0:
                continue
            # Only compare consecutive hours (skip if gap > 2 hours)
            try:
                t_prev = datetime.strptime(sorted_results[idx - 1]["date"], "%Y-%m-%d %H:%M:%S")
                t_curr = datetime.strptime(sorted_results[idx]["date"], "%Y-%m-%d %H:%M:%S")
                if (t_curr - t_prev).total_seconds() > 2 * 3600:
                    continue
            except (ValueError, TypeError, KeyError):
                continue
            actual_move_pct = abs(curr_actual - prev_actual) / prev_actual * 100
            directional_total += 1
            if actual_move_pct < _DIR_NEUTRAL_PCT:
                # Market was flat — count as correct regardless of prediction
                directional_correct += 1
            else:
                pred_dir = 1 if curr_predicted >= prev_actual else -1
                actual_dir = 1 if curr_actual >= prev_actual else -1
                if pred_dir == actual_dir:
                    directional_correct += 1

        result = {
            "plan_generated_at": generated_at,
            "evaluated_at": iso_now_ist(),
            "current_price_at_prediction": round(current_price, 2),
            "days_evaluated": len(evaluated_days),
            "days_total": len(preds),
            "mae": round(np.mean(errors), 2),
            "mape": round(np.mean(pct_errors), 2),
            "max_error": round(max(errors), 2),
            "band_hit_rate": round(sum(band_hits) / len(band_hits) * 100, 1),
            "directional_accuracy": round(
                directional_correct / directional_total * 100, 1
            ) if directional_total > 0 else 50.0,
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
        # Cap accuracy log to _MAX_LOG_ENTRIES entries (trim oldest)
        if len(self._log) > self._MAX_LOG_ENTRIES:
            self._log = self._log[-self._MAX_LOG_ENTRIES:]
        self._save_log()

        # Also persist to the permanent prediction archive (no cap)
        self._archive_evaluation(result)

        return result

    def _get_actual_price(self, close_series: pd.Series, target_date) -> Optional[float]:
        """Get the close for a target hour, using nearest-match with fallback.

        First tries nearest candle within 90 minutes (handles the 30-minute
        IST offset from UTC candle boundaries).  Falls back to stepping
        backwards hour-by-hour for up to 72 hours to handle weekends and
        market holidays (e.g., Good Friday, Christmas).
        """
        if close_series.empty:
            return None

        target = pd.Timestamp(target_date)

        # 1. Try nearest candle within 90 minutes
        try:
            diffs = abs(close_series.index - target)
            nearest_pos = diffs.argmin()
            if diffs[nearest_pos] <= pd.Timedelta(minutes=90):
                val = close_series.iloc[nearest_pos]
                if pd.notna(val):
                    return float(val)
        except Exception:
            pass

        # 2. Fallback: step back hour-by-hour (weekends/holidays)
        for offset in range(0, 73):
            check = target_date - timedelta(hours=offset)
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

    def get_aggregate_stats(self, recent_hours: int = 72) -> Optional[dict]:
        """Compute aggregate accuracy over recent predictions.

        Parameters
        ----------
        recent_hours : int
            Only include evaluated hours whose predicted timestamp falls
            within the last *recent_hours* hours.  Defaults to 72 (3 days)
            so that metrics reflect recent model performance instead of
            being diluted by hundreds of old, potentially poor predictions.

        Deduplication: when multiple plans predict the same hour, the
        prediction from the plan generated **most recently before** that
        hour is used.  This is the fairest approach because that plan had
        the freshest market information at generation time.

        Directional accuracy is computed hour-over-hour from the
        deduplicated series (consistent with MAPE / MAE / Band Hit).
        """
        if not self._log:
            return None

        now_ts = now_ist().replace(minute=0, second=0, microsecond=0).replace(tzinfo=None)
        cutoff = now_ts - timedelta(hours=recent_hours)

        # Collect all results keyed by predicted hour.
        # For dedup: use the prediction from the most recently generated
        # plan (closest to but before the predicted hour).  This avoids
        # cherry-picking the best result from overlapping forecasts.
        by_date: dict = {}
        for ev in self._log:
            plan_gen = ev.get("plan_generated_at", "")
            for d in ev.get("daily_results", []):
                date_key = d.get("date", "")

                # Filter to recent window
                try:
                    d_ts = datetime.strptime(date_key, "%Y-%m-%d %H:%M:%S")
                except (ValueError, TypeError):
                    continue
                if d_ts < cutoff:
                    continue

                # Determine which plan generated this prediction
                gen_at = d.get("plan_generated_at", plan_gen)

                # Pick the most recently generated plan for each hour.
                # Parse generation timestamps; prefer the one closest to
                # (but before) the predicted hour.
                if date_key in by_date:
                    old = by_date[date_key]
                    old_gen = old.get("plan_generated_at", "")
                    try:
                        new_gen_dt = datetime.fromisoformat(str(gen_at)).replace(tzinfo=None)
                        old_gen_dt = datetime.fromisoformat(str(old_gen)).replace(tzinfo=None)
                        # Prefer more recent plan (closer to the predicted hour)
                        if new_gen_dt > old_gen_dt:
                            by_date[date_key] = d
                    except (ValueError, TypeError):
                        pass  # keep existing if we can't parse
                else:
                    by_date[date_key] = d

        all_days = list(by_date.values())

        if not all_days:
            return None

        errors = [abs(d.get("error", 0)) for d in all_days]
        pct_errors = [d["pct_error"] for d in all_days]
        band_hits = [d["within_band"] for d in all_days]

        # Directional accuracy: hour-over-hour from the deduplicated series.
        # Sort by time, then for each consecutive pair check if the predicted
        # direction matches the actual direction of price movement.
        # Neutral zone: if actual moved < 0.05%, count as correct (market flat).
        sorted_days = sorted(all_days, key=lambda d: d.get("date", ""))
        dir_correct = 0
        dir_total = 0
        for i in range(1, len(sorted_days)):
            prev_actual = sorted_days[i - 1].get("actual", 0)
            curr_predicted = sorted_days[i].get("predicted", 0)
            curr_actual = sorted_days[i].get("actual", 0)
            if prev_actual <= 0:
                continue
            # Only compare consecutive hours (skip if gap exceeds 2 hours)
            try:
                t_prev = datetime.strptime(sorted_days[i - 1]["date"], "%Y-%m-%d %H:%M:%S")
                t_curr = datetime.strptime(sorted_days[i]["date"], "%Y-%m-%d %H:%M:%S")
                if (t_curr - t_prev).total_seconds() > 2 * 3600:
                    continue
            except (ValueError, TypeError, KeyError):
                continue
            actual_move_pct = abs(curr_actual - prev_actual) / prev_actual * 100
            dir_total += 1
            if actual_move_pct < _DIR_NEUTRAL_PCT:
                # Market was flat — count as correct regardless
                dir_correct += 1
            else:
                pred_dir = 1 if curr_predicted >= prev_actual else -1
                actual_dir = 1 if curr_actual >= prev_actual else -1
                if pred_dir == actual_dir:
                    dir_correct += 1

        # Default to 50% (random chance) when no consecutive pairs are available
        directional_accuracy = round(
            dir_correct / dir_total * 100, 1
        ) if dir_total > 0 else 50.0

        # Count unique calendar dates covered by the evaluated hours
        unique_dates = set()
        for d in all_days:
            try:
                unique_dates.add(datetime.strptime(d["date"], "%Y-%m-%d %H:%M:%S").date())
            except (ValueError, TypeError, KeyError):
                pass

        # Count total distinct hourly predictions across the FULL log
        # (not limited to the recent_hours window) so "Unique Hours"
        # reflects all hours we have ever predicted and scored.
        all_hours_set: set[str] = set()
        all_dates_set: set[datetime.date] = set()
        for ev in self._log:
            for d in ev.get("daily_results", []):
                date_key = d.get("date", "")
                if date_key:
                    all_hours_set.add(date_key)
                    try:
                        all_dates_set.add(datetime.strptime(date_key, "%Y-%m-%d %H:%M:%S").date())
                    except (ValueError, TypeError):
                        pass

        return {
            "total_predictions_evaluated": len(all_days),
            "unique_dates_evaluated": len(unique_dates),
            "total_plans_evaluated": len(self._log),
            "overall_mae": round(np.mean(errors), 2),
            "overall_mape": round(np.mean(pct_errors), 2),
            "overall_band_hit_rate": round(sum(band_hits) / len(band_hits) * 100, 1),
            "best_mae": round(min(r["mae"] for r in self._log), 2),
            "worst_mae": round(max(r["mae"] for r in self._log), 2),
            "avg_directional_accuracy": directional_accuracy,
            "total_unique_hours": len(all_hours_set),
            "total_unique_dates": len(all_dates_set),
        }

    # ── Refresh: re-evaluate ALL stored plans against latest data ────

    def refresh_all(self) -> int:
        """
        Re-evaluate every stored plan against the latest actual prices.
        Called automatically when market data updates (new day closes).
        Returns the number of plans that had new evaluable days.
        """
        # On Cloud, local cache may have been wiped while the cached object
        # is still alive.  Re-load from disk/Gist if in-memory state is empty.
        if not self._stored_plans:
            self._stored_plans = self._load_stored_plans()
        if not self._log:
            self._log = self._load_log()
            self._purge_stale_entries()

        updated = 0
        for plan_dict in self._stored_plans:
            before_count = self._evaluable_day_count(plan_dict.get("generated_at", ""))
            self.evaluate_plan(plan_dict)
            after_count = self._evaluable_day_count(plan_dict.get("generated_at", ""))
            if after_count > before_count:
                updated += 1

        self._last_checked = iso_now_ist()
        logger.info(f"Accuracy refresh complete — {updated} plans had new data")
        return updated

    def _evaluable_day_count(self, generated_at: str) -> int:
        for r in self._log:
            if r.get("plan_generated_at") == generated_at:
                return r.get("days_evaluated", 0)
        return 0

    @property
    def last_checked(self) -> Optional[str]:
        return self._last_checked

    # ── Background auto-check thread ─────────────────────────────────

    def start_auto_check(self, interval_hours: float = 6):
        """
        Start a background thread that re-evaluates stored predictions
        periodically (default every 6 hours).  This catches new day-close
        data as it becomes available without requiring user interaction.
        """
        if self._auto_running:
            return
        self._auto_running = True

        def _loop():
            # Initial check on startup
            try:
                self.refresh_all()
            except Exception as e:
                logger.warning(f"Initial accuracy refresh failed: {e}")

            while self._auto_running:
                time.sleep(interval_hours * 3600)
                try:
                    self.refresh_all()
                except Exception as e:
                    logger.warning(f"Auto accuracy refresh failed: {e}")

        t = threading.Thread(target=_loop, daemon=True, name="accuracy-auto-check")
        t.start()
        logger.info(f"Accuracy auto-check started — refreshing every {interval_hours}h")

    def stop_auto_check(self):
        self._auto_running = False
