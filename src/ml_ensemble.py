"""
ML Ensemble – production-grade price prediction using stacked models.

Architecture
============
Layer 1 (base learners):
  - XGBoost (gradient boosting)
  - LightGBM (gradient boosting, leaf-wise)
  - Ridge Regression (linear baseline)

Layer 2 (meta-learner):
  - Ridge stacks the three base predictions → final point estimate

Confidence bands:
  - XGBoost quantile regression (alpha=0.10, 0.90) → 80 % prediction interval

Explainability:
  - SHAP TreeExplainer on the XGBoost base model → per-feature contributions

All prices are internally USD/oz.  Conversion to INR/10g happens once at the end.
"""

from __future__ import annotations

import json
import math
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.linear_model import Ridge

from .config import CACHE_DIR, PREDICTION_HOURS
from .data_fetchers.market_data import MarketDataFetcher
from .guardrails import validate_xgb_predictions, validate_price_series, MAX_HOURLY_MOVE_PCT
from .residual_learner import ResidualLearner

# ── Feature engineering constants ────────────────────────────────────
_LAGS = [1, 2, 3, 6, 12, 24]
_ROLL_WINDOWS = [6, 12, 24]
_MIN_HISTORY = 25            # minimum samples for feature construction
_TRAIN_PERIOD_DAYS = 90      # 3 months of hourly data (~2160 bars)
_MIN_TRAIN_SAMPLES = 200     # hard floor after cleanup

# ── Internal feature names (used for model training) ─────────────────
FEATURE_NAMES = (
    [f"lag_{l}" for l in _LAGS]
    + [f"roll_{w}" for w in _ROLL_WINDOWS]
    + ["ret_1h", "ret_6h", "vol_12"]
    + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    + [
        "sentiment_score",
        "geopolitical_risk",
        "macro_outlook",
        "technical_signal",
        "etf_flow_signal",
        "oil_energy_signal",
        "historical_seasonal",
        "trend_strength",
    ]
)

# ── Human-friendly display names for dashboard / SHAP charts ─────────
FEATURE_DISPLAY_NAMES: dict[str, str] = {
    "lag_1":              "Price 1 Hour Ago",
    "lag_2":              "Price 2 Hours Ago",
    "lag_3":              "Price 3 Hours Ago",
    "lag_6":              "Price 6 Hours Ago",
    "lag_12":             "Price 12 Hours Ago",
    "lag_24":             "Price 24 Hours Ago",
    "roll_6":             "6-Hour Average Price",
    "roll_12":            "12-Hour Average Price",
    "roll_24":            "24-Hour Average Price",
    "ret_1h":             "1-Hour Price Change %",
    "ret_6h":             "6-Hour Price Change %",
    "vol_12":             "12-Hour Volatility",
    "hour_sin":           "Time of Day (sine)",
    "hour_cos":           "Time of Day (cosine)",
    "dow_sin":            "Day of Week (sine)",
    "dow_cos":            "Day of Week (cosine)",
    "sentiment_score":    "Market Sentiment",
    "geopolitical_risk":  "Geopolitical Risk",
    "macro_outlook":      "Macro-Economic Outlook",
    "technical_signal":   "Technical Analysis Signal",
    "etf_flow_signal":    "ETF Fund Flows",
    "oil_energy_signal":  "Oil & Energy Impact",
    "historical_seasonal":"Seasonal Pattern",
    "trend_strength":     "Trend Strength",
}


def _build_feature_vector(
    history: list[float],
    timestamp: Optional[pd.Timestamp] = None,
    agent_signals: Optional[dict[str, float]] = None,
) -> np.ndarray:
    """Build one feature row from price history + optional agent signals."""
    arr = np.asarray(history, dtype=np.float64)
    feats: list[float] = []

    # Lags
    for l in _LAGS:
        feats.append(arr[-l] if l <= len(arr) else arr[0])

    # Rolling means
    for w in _ROLL_WINDOWS:
        feats.append(float(np.mean(arr[-w:])) if w <= len(arr) else float(arr[-1]))

    # Returns
    lag1 = arr[-1]
    lag2 = arr[-2] if len(arr) > 1 else arr[-1]
    lag6 = arr[-6] if len(arr) > 5 else arr[0]
    feats.append(0.0 if lag2 == 0 else (lag1 - lag2) / lag2)  # ret_1h
    feats.append(0.0 if lag6 == 0 else (lag1 - lag6) / lag6)  # ret_6h

    # Volatility (12h)
    if len(arr) > 13:
        feats.append(float(np.std(np.diff(arr[-13:]) / np.clip(arr[-13:-1], 1e-9, None))))
    else:
        feats.append(0.002)

    # Time-of-day cyclical features
    if timestamp is not None:
        hour = timestamp.hour + timestamp.minute / 60.0
        dow = timestamp.dayofweek
    else:
        hour, dow = 12.0, 2  # midweek noon fallback
    feats.append(math.sin(2 * math.pi * hour / 24))
    feats.append(math.cos(2 * math.pi * hour / 24))
    feats.append(math.sin(2 * math.pi * dow / 7))
    feats.append(math.cos(2 * math.pi * dow / 7))

    # Agent-derived signals (numeric features extracted by LLM agents)
    sig = agent_signals or {}
    feats.append(sig.get("sentiment_score", 0.0))
    feats.append(sig.get("geopolitical_risk", 0.0))
    feats.append(sig.get("macro_outlook", 0.0))
    feats.append(sig.get("technical_signal", 0.0))
    feats.append(sig.get("etf_flow_signal", 0.0))
    feats.append(sig.get("oil_energy_signal", 0.0))
    feats.append(sig.get("historical_seasonal", 0.0))
    feats.append(sig.get("trend_strength", 0.0))

    return np.array(feats, dtype=np.float64)


class MLEnsemble:
    """Three-model stacking ensemble with quantile bands and SHAP."""

    def __init__(self):
        self._market = MarketDataFetcher()
        self._residual_learner = ResidualLearner()
        self._xgb_model = None
        self._lgb_model = None
        self._ridge_model = None
        self._meta_model = None
        self._xgb_lo = None      # quantile α=0.10
        self._xgb_hi = None      # quantile α=0.90
        self._is_trained = False
        self._shap_values: Optional[np.ndarray] = None
        self._last_X_pred: Optional[np.ndarray] = None

    # ── Training ─────────────────────────────────────────────────────

    def train(self, agent_signals: Optional[dict[str, float]] = None) -> bool:
        """
        Fetch data, build features, train all base + meta + quantile models.
        Returns True on success.
        """
        try:
            from xgboost import XGBRegressor
            import lightgbm as lgb
        except ImportError as e:
            logger.error(f"ML dependency missing: {e}")
            return False

        # 1. Fetch hourly gold data
        df = self._market.fetch_ticker("GC=F", period_days=_TRAIN_PERIOD_DAYS, interval="1h")
        if df.empty or "Close" not in df:
            logger.warning("ML ensemble: no training data")
            return False

        close = pd.to_numeric(df["Close"].squeeze(), errors="coerce").dropna()
        if len(close) < _MIN_TRAIN_SAMPLES:
            logger.warning(f"ML ensemble: only {len(close)} samples (need {_MIN_TRAIN_SAMPLES})")
            return False

        values_list = validate_price_series(close.to_numpy(dtype=float).tolist(), "ensemble_train")
        if len(values_list) < _MIN_TRAIN_SAMPLES:
            return False
        values = np.array(values_list, dtype=float)
        timestamps = close.index

        # 2. Build feature matrix
        X_rows, y_rows = [], []
        for i in range(_MIN_HISTORY, len(values)):
            hist = values[:i].tolist()
            ts = timestamps[i] if i < len(timestamps) else None
            X_rows.append(_build_feature_vector(hist, ts, agent_signals))
            y_rows.append(float(values[i]))

        if len(X_rows) < 100:
            return False

        X = np.vstack(X_rows)
        y = np.array(y_rows, dtype=float)

        # 3. Train/test split (last 20% for meta-learner training)
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

        # 4. Base learners
        self._xgb_model = XGBRegressor(
            n_estimators=300, max_depth=5, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, objective="reg:squarederror",
        )
        self._xgb_model.fit(X_train, y_train)

        self._lgb_model = lgb.LGBMRegressor(
            n_estimators=300, max_depth=6, learning_rate=0.05,
            subsample=0.85, colsample_bytree=0.85,
            reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbose=-1,
        )
        df_train = pd.DataFrame(X_train, columns=FEATURE_NAMES)
        self._lgb_model.fit(df_train, y_train)

        self._ridge_model = Ridge(alpha=1.0)
        self._ridge_model.fit(X_train, y_train)

        # 5. Quantile models for confidence bands (10th, 90th percentile)
        self._xgb_lo = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.85, random_state=42,
            objective="reg:quantileerror", quantile_alpha=0.10,
        )
        self._xgb_lo.fit(X_train, y_train)

        self._xgb_hi = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.85, random_state=42,
            objective="reg:quantileerror", quantile_alpha=0.90,
        )
        self._xgb_hi.fit(X_train, y_train)

        # 6. Meta-learner: stack base predictions on validation set
        p_xgb = self._xgb_model.predict(X_val)
        p_lgb = self._lgb_model.predict(pd.DataFrame(X_val, columns=FEATURE_NAMES))
        p_ridge = self._ridge_model.predict(X_val)
        stack_X = np.column_stack([p_xgb, p_lgb, p_ridge])

        self._meta_model = Ridge(alpha=0.5)
        self._meta_model.fit(stack_X, y_val)

        self._is_trained = True
        logger.info(
            f"ML ensemble trained: {len(X_train)} train, {len(X_val)} val samples, "
            f"{len(FEATURE_NAMES)} features"
        )
        return True

    # ── Prediction ───────────────────────────────────────────────────

    def predict(
        self,
        agent_signals: Optional[dict[str, float]] = None,
    ) -> list[dict]:
        """
        Forecast the next PREDICTION_HOURS hourly prices in INR/10g.

        Returns list of dicts with keys:
            date, predicted_price, low_range, high_range,
            xgb_price, lgb_price, ridge_price (component prices for transparency)
        """
        if not self._is_trained:
            logger.warning("ML ensemble: not trained, cannot predict")
            return []

        try:
            df = self._market.fetch_ticker("GC=F", period_days=30, interval="1h")
            if df.empty:
                return []

            close = pd.to_numeric(df["Close"].squeeze(), errors="coerce").dropna()
            if len(close) < _MIN_HISTORY:
                return []

            values = close.to_numpy(dtype=float)
            history = values.tolist()
            last_ts = close.index.max()

            usdinr = self._market.get_usdinr_rate()
            oz_to_10g = 10.0 / 31.1035

            preds: list[dict] = []
            X_for_shap: list[np.ndarray] = []

            # Track the previous USD price so we can apply a per-step cap
            # before feeding predictions back into the auto-regressive history.
            # Without this cap, one bad step contaminates every subsequent lag
            # feature and causes exponentially growing errors.
            prev_step_usd = float(values[-1]) if len(values) > 0 else None

            for h in range(PREDICTION_HOURS):
                ts = pd.Timestamp(last_ts) + pd.Timedelta(hours=h + 1)
                feat = _build_feature_vector(history, ts, agent_signals).reshape(1, -1)

                # Base predictions (USD/oz)
                p_xgb = float(self._xgb_model.predict(feat)[0])
                p_lgb = float(self._lgb_model.predict(pd.DataFrame(feat, columns=FEATURE_NAMES))[0])
                p_ridge = float(self._ridge_model.predict(feat)[0])

                # Meta-learner stacked prediction
                stack = np.array([[p_xgb, p_lgb, p_ridge]])
                p_final_usd = float(self._meta_model.predict(stack)[0])

                # ── Auto-regressive guardrail (USD space) ────────────────
                # Cap the USD prediction to ±MAX_HOURLY_MOVE_PCT% of the
                # previous USD price *before* appending to history.  This
                # prevents a single bad step from corrupting all subsequent
                # lag features and causing exponential error amplification.
                if prev_step_usd is not None and prev_step_usd > 0 and math.isfinite(p_final_usd):
                    move_pct = (p_final_usd - prev_step_usd) / prev_step_usd * 100
                    if abs(move_pct) > MAX_HOURLY_MOVE_PCT:
                        direction = 1 if p_final_usd > prev_step_usd else -1
                        p_final_usd = prev_step_usd * (1 + direction * MAX_HOURLY_MOVE_PCT / 100)
                        logger.debug(
                            f"AR guardrail (step {h+1}): USD move {move_pct:.1f}% "
                            f"capped to {direction * MAX_HOURLY_MOVE_PCT:.1f}%"
                        )

                # Quantile bands (USD/oz)
                lo_usd = float(self._xgb_lo.predict(feat)[0])
                hi_usd = float(self._xgb_hi.predict(feat)[0])

                # Convert to INR/10g
                p_final_inr = p_final_usd * usdinr * oz_to_10g
                lo_inr = lo_usd * usdinr * oz_to_10g
                hi_inr = hi_usd * usdinr * oz_to_10g

                # Ensure band ordering
                lo_inr, hi_inr = min(lo_inr, hi_inr), max(lo_inr, hi_inr)
                if lo_inr > p_final_inr:
                    lo_inr = p_final_inr * 0.997
                if hi_inr < p_final_inr:
                    hi_inr = p_final_inr * 1.003

                preds.append({
                    "date": ts.strftime("%Y-%m-%d %H:00"),
                    "xgb_price": round(p_final_inr, 2),
                    "xgb_low": round(lo_inr, 2),
                    "xgb_high": round(hi_inr, 2),
                    # Component prices for UI transparency
                    "component_xgb": round(p_xgb * usdinr * oz_to_10g, 2),
                    "component_lgb": round(p_lgb * usdinr * oz_to_10g, 2),
                    "component_ridge": round(p_ridge * usdinr * oz_to_10g, 2),
                })
                X_for_shap.append(feat.flatten())

                # Feed the *capped* USD price back into history so all
                # subsequent lag features stay in a plausible range.
                history.append(p_final_usd)
                prev_step_usd = p_final_usd

            # Validate predictions
            current_inr = self._market.get_gold_inr_price()
            if not (isinstance(current_inr, (int, float)) and np.isfinite(current_inr) and current_inr > 0):
                current_inr = preds[0]["xgb_price"] if preds else 70000.0
            preds = validate_xgb_predictions(preds, current_inr)

            # Apply residual corrections from past errors
            preds = self._residual_learner.correct_predictions(preds)

            # Store for SHAP computation
            self._last_X_pred = np.vstack(X_for_shap) if X_for_shap else None

            return preds

        except Exception as e:
            logger.error(f"ML ensemble prediction failed: {e}")
            return []

    # ── SHAP Explainability ──────────────────────────────────────────

    def get_shap_explanation(self) -> Optional[dict]:
        """
        Compute SHAP values for the most recent prediction.
        Returns dict with feature importances and per-hour breakdowns.
        """
        if self._xgb_model is None or self._last_X_pred is None:
            return None

        try:
            import shap
            explainer = shap.TreeExplainer(self._xgb_model)
            shap_values = explainer.shap_values(self._last_X_pred)
            self._shap_values = shap_values

            # Average absolute SHAP across all prediction hours
            mean_abs_shap = np.mean(np.abs(shap_values), axis=0)

            # Build feature importance ranking
            n_features = min(len(FEATURE_NAMES), len(mean_abs_shap))
            importance = []
            for i in range(n_features):
                raw_name = FEATURE_NAMES[i]
                importance.append({
                    "feature": FEATURE_DISPLAY_NAMES.get(raw_name, raw_name),
                    "importance": round(float(mean_abs_shap[i]), 4),
                })
            importance.sort(key=lambda x: x["importance"], reverse=True)

            # Per-hour top 3 drivers
            hourly_drivers = []
            for h in range(min(PREDICTION_HOURS, len(shap_values))):
                h_shap = shap_values[h]
                top_idx = np.argsort(np.abs(h_shap))[-3:][::-1]
                drivers = []
                for idx in top_idx:
                    if idx < n_features:
                        raw_name = FEATURE_NAMES[idx]
                        display_name = FEATURE_DISPLAY_NAMES.get(raw_name, raw_name)
                        direction = "↑" if h_shap[idx] > 0 else "↓"
                        drivers.append({
                            "name": display_name,
                            "direction": direction,
                            "value": round(float(h_shap[idx]), 2),
                        })
                hourly_drivers.append({
                    "hour": h + 1,
                    "drivers": drivers,
                })

            return {
                "feature_importance": importance[:15],
                "hourly_drivers": hourly_drivers,
                "model_type": "XGBoost + LightGBM + Ridge (stacked)",
                "total_features": n_features,
            }

        except ImportError:
            logger.warning("shap package not installed – skipping explainability")
            return None
        except Exception as e:
            logger.warning(f"SHAP computation failed: {e}")
            return None

    # ── Residual learning ────────────────────────────────────────────

    def update_residuals(self, accuracy_log: list[dict]):
        """Feed accuracy data to the residual learner."""
        self._residual_learner.update_from_accuracy_log(accuracy_log)

    def get_residual_summary(self) -> Optional[dict]:
        return self._residual_learner.get_correction_summary()
