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
from .guardrails import validate_xgb_predictions, validate_price_series, _band_envelope_pct
from .residual_learner import ResidualLearner

# ── Feature engineering constants ────────────────────────────────────
_LAGS = [1, 2, 3, 6, 12, 24]
_ROLL_WINDOWS = [6, 12, 24]
_MIN_HISTORY = 25            # minimum samples for feature construction
_TRAIN_PERIOD_DAYS = 90      # 3 months of hourly data (~2160 bars)
_MIN_TRAIN_SAMPLES = 200     # hard floor after cleanup

# ── Internal feature names (used for model training) ─────────────────
# NOTE: Only 16 time-series features are used for training, because agent
# signals are a static snapshot (same value for every historical row) and
# therefore carry zero information for the tree models.  Agent signals are
# applied as a *post-prediction* directional adjustment instead.
FEATURE_NAMES = (
    [f"lag_{l}" for l in _LAGS]
    + [f"roll_{w}" for w in _ROLL_WINDOWS]
    + ["ret_1h", "ret_6h", "vol_12"]
    + ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]
)

# The 8 agent-derived signal names (used for post-prediction adjustment)
AGENT_SIGNAL_NAMES = [
    "sentiment_score",
    "geopolitical_risk",
    "macro_outlook",
    "technical_signal",
    "etf_flow_signal",
    "oil_energy_signal",
    "historical_seasonal",
    "trend_strength",
]

# Weights for how strongly each agent signal influences the final price.
# Higher weight = more influence.  Sum of weights doesn't need to be 1;
# they are normalised internally.
_AGENT_WEIGHTS: dict[str, float] = {
    "sentiment_score":    1.0,
    "geopolitical_risk":  0.8,
    "macro_outlook":      1.0,
    "technical_signal":   1.2,   # technical analysis has strong short-term signal
    "etf_flow_signal":    0.7,
    "oil_energy_signal":  0.6,
    "historical_seasonal":0.5,
    "trend_strength":     1.0,
}

# Maximum % adjustment agents can apply to the base ML prediction.
# Reduced from 2.5% to 1.5%: larger values cause systematic directional
# bias that compounds through the autoregressive loop.
_MAX_AGENT_ADJUSTMENT_PCT = 1.5

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
) -> np.ndarray:
    """Build one feature row from price history (16 time-series features).

    Agent signals are NOT included here because they are a static snapshot
    (identical for every historical training row) and would carry zero
    information for the tree models.  They are applied as a post-prediction
    directional adjustment instead — see ``_compute_agent_adjustment``.
    """
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

    return np.array(feats, dtype=np.float64)


def _compute_agent_adjustment(
    agent_signals: Optional[dict[str, float]],
) -> float:
    """Compute a directional adjustment factor from agent signals.

    Returns a multiplier in the range ``1 - MAX_ADJ … 1 + MAX_ADJ``
    (e.g. 0.985 … 1.015 for ±1.5 %).

    The composite signal is a weighted average of the 8 agent scores,
    each in [-1, +1], mapped to a bounded percentage adjustment.
    """
    if not agent_signals:
        return 1.0

    weighted_sum = 0.0
    total_weight = 0.0
    for name, weight in _AGENT_WEIGHTS.items():
        val = agent_signals.get(name, 0.0)
        weighted_sum += val * weight
        total_weight += weight

    if total_weight == 0:
        return 1.0

    # Composite signal in [-1, +1]
    composite = max(-1.0, min(1.0, weighted_sum / total_weight))

    # Map to a bounded percentage adjustment
    adjustment_pct = composite * _MAX_AGENT_ADJUSTMENT_PCT / 100.0
    return 1.0 + adjustment_pct


class MLEnsemble:
    """Three-model stacking ensemble with quantile bands and SHAP."""

    def __init__(self):
        self._market = MarketDataFetcher()
        self._residual_learner = ResidualLearner()
        self._xgb_model = None
        self._lgb_model = None
        self._ridge_model = None
        self._meta_model = None
        self._xgb_lo = None      # quantile α=0.05
        self._xgb_hi = None      # quantile α=0.95
        self._is_trained = False
        self._shap_values: Optional[np.ndarray] = None
        self._last_X_pred: Optional[np.ndarray] = None

    # ── Training ─────────────────────────────────────────────────────

    def train(self, agent_signals: Optional[dict[str, float]] = None) -> bool:
        """
        Fetch data, build features, train all base + meta + quantile models.
        Returns True on success.

        ``agent_signals`` is accepted for interface compatibility but is NOT
        used during training — agent signals are constant across all
        historical rows and carry zero information for tree models.
        They are applied as a post-prediction adjustment in ``predict()``.
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

        # 2. Build feature matrix (16 time-series features only, no agent signals)
        X_rows, y_rows = [], []
        for i in range(_MIN_HISTORY, len(values)):
            hist = values[:i].tolist()
            ts = timestamps[i] if i < len(timestamps) else None
            X_rows.append(_build_feature_vector(hist, ts))
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

        # 5. Quantile models for confidence bands (5th, 95th percentile)
        #    Using a wider 90% prediction interval (was 80%) so actual
        #    prices fall within the band more often → higher band hit rate.
        self._xgb_lo = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.85, random_state=42,
            objective="reg:quantileerror", quantile_alpha=0.05,
        )
        self._xgb_lo.fit(X_train, y_train)

        self._xgb_hi = XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.85, random_state=42,
            objective="reg:quantileerror", quantile_alpha=0.95,
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

        The base forecast comes from the ML ensemble (16 time-series features).
        Agent signals are applied as a **post-prediction directional adjustment**
        so the 8 specialist agents genuinely influence the final price.

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

            # Compute agent adjustment multiplier (applied after ML prediction)
            agent_multiplier = _compute_agent_adjustment(agent_signals)
            agent_adj_pct = (agent_multiplier - 1.0) * 100
            logger.info(
                f"Agent signal adjustment: {agent_adj_pct:+.3f}% "
                f"(multiplier={agent_multiplier:.5f}, signals={agent_signals})"
            )

            preds: list[dict] = []
            X_for_shap: list[np.ndarray] = []

            # Reference USD price for sanity checks in the auto-regressive loop
            ref_usd_price = float(values[-1])  # last known actual USD/oz price

            for h in range(PREDICTION_HOURS):
                ts = pd.Timestamp(last_ts) + pd.Timedelta(hours=h + 1)
                feat = _build_feature_vector(history, ts).reshape(1, -1)

                # Base predictions (USD/oz)
                p_xgb = float(self._xgb_model.predict(feat)[0])
                p_lgb = float(self._lgb_model.predict(pd.DataFrame(feat, columns=FEATURE_NAMES))[0])
                p_ridge = float(self._ridge_model.predict(feat)[0])

                # Meta-learner stacked prediction
                stack = np.array([[p_xgb, p_lgb, p_ridge]])
                p_final_usd = float(self._meta_model.predict(stack)[0])

                # ── Apply agent signal adjustment (directional bias) ──
                # This is where the 8 specialist agents actually influence the
                # predicted price.  The adjustment decays slightly for later
                # hours (agents have strongest signal for near-term).
                horizon_decay = max(0.3, 1.0 - 0.03 * h)  # hour 0→1.0, hour 24→0.30
                effective_multiplier = 1.0 + (agent_multiplier - 1.0) * horizon_decay
                p_final_usd *= effective_multiplier

                # ── Mean-reversion pressure ──
                # Prevent autoregressive drift by blending the prediction with
                # the reference price.  The blend increases for later horizons
                # so near-term predictions stay responsive while far-horizon
                # ones don't compound into unrealistic trends.
                # Uses 0.04*h to ramp faster; capped at 50% so the last few
                # hours still partially reflect model direction.
                reversion_strength = min(0.50, 0.04 * h)  # hour 1→4%, hour 6→24%, hour 12→48%, hour 13+→50%
                p_final_usd = p_final_usd * (1.0 - reversion_strength) + ref_usd_price * reversion_strength

                # ── Per-step USD sanity check ──
                # Prevent runaway drift: cap total deviation and per-step moves.
                # Total cap ±3%: gold rarely moves >2-3% in a full day.
                if ref_usd_price > 0:
                    max_total_dev = ref_usd_price * 0.03  # ±3% total
                    p_final_usd = max(ref_usd_price - max_total_dev,
                                      min(p_final_usd, ref_usd_price + max_total_dev))
                prev_usd = history[-1] if history else ref_usd_price
                if prev_usd > 0:
                    # Tight step caps to prevent compounding:
                    # 0.5% h1-6, 0.35% h7-12, 0.25% h13+
                    # Theoretical max from steps alone: 6×0.5% + 6×0.35% + 12×0.25% = 8.1%
                    # but total cap (3%) catches runaway much earlier.
                    if h < 6:
                        step_pct = 0.005
                    elif h < 12:
                        step_pct = 0.0035
                    else:
                        step_pct = 0.0025
                    max_step = prev_usd * step_pct
                    p_final_usd = max(prev_usd - max_step,
                                      min(p_final_usd, prev_usd + max_step))

                # Quantile bands (USD/oz)
                lo_usd = float(self._xgb_lo.predict(feat)[0])
                hi_usd = float(self._xgb_hi.predict(feat)[0])

                # ── Progressive band widening ──
                # Uncertainty grows with forecast distance.  Widen the raw
                # quantile band by sqrt(h+1).  A separate minimum floor
                # (also sqrt-based) prevents near-hour bands from collapsing.
                band_half = (hi_usd - lo_usd) / 2.0
                widen_factor = math.sqrt(h + 1)
                band_half *= widen_factor
                # Minimum floor: 0.3% of reference × sqrt(h+1)
                min_band_half = ref_usd_price * 0.003 * widen_factor
                band_half = max(band_half, min_band_half)
                # Re-center bands around the predicted price (not the quantile center)
                lo_usd = p_final_usd - band_half
                hi_usd = p_final_usd + band_half

                # Clamp quantile bands to a horizon-aware deviation envelope
                # centred on the *predicted* price (not the reference price).
                # This keeps bands symmetric around the prediction and avoids
                # the one-sided collapse seen when the prediction is near the
                # total deviation cap.
                if ref_usd_price > 0:
                    max_dev_pct = _band_envelope_pct(h)
                    max_dev_abs = ref_usd_price * max_dev_pct
                    lo_usd = max(p_final_usd - max_dev_abs, lo_usd)
                    hi_usd = min(p_final_usd + max_dev_abs, hi_usd)

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

                # Auto-regressive: feed predicted USD price back into history
                history.append(p_final_usd)

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

    def get_shap_explanation(self, agent_signals: Optional[dict[str, float]] = None) -> Optional[dict]:
        """
        Compute SHAP values for the most recent prediction.
        Returns dict with feature importances, per-hour breakdowns,
        and agent signal contribution breakdown.
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

            # Add agent signal contributions as pseudo-importance entries
            # so the UI shows them alongside SHAP features.
            # Scale to USD terms (matching SHAP value units) using the
            # reference price from the most recent lag_1 feature.
            if agent_signals and self._last_X_pred is not None and len(self._last_X_pred) > 0:
                agent_multiplier = _compute_agent_adjustment(agent_signals)
                ref_price = float(self._last_X_pred[0][0])  # lag_1 = latest price (USD/oz)
                agent_dollar_impact = abs(agent_multiplier - 1.0) * ref_price
                total_weight = sum(_AGENT_WEIGHTS.values())
                for name in AGENT_SIGNAL_NAMES:
                    val = agent_signals.get(name, 0.0)
                    weight = _AGENT_WEIGHTS.get(name, 0.5)
                    # Contribution in USD: proportional to |signal × weight|
                    # relative to total agent dollar impact on the prediction.
                    contribution = abs(val) * weight / total_weight * agent_dollar_impact
                    if contribution < 0.001:
                        continue  # skip agents with negligible contribution
                    display_name = FEATURE_DISPLAY_NAMES.get(name, name)
                    importance.append({
                        "feature": f"{display_name} (Agent)",
                        "importance": round(contribution, 4),
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

            # Agent signal summary for transparency
            agent_summary = None
            if agent_signals:
                agent_multiplier = _compute_agent_adjustment(agent_signals)
                agent_summary = {
                    "composite_adjustment_pct": round((agent_multiplier - 1.0) * 100, 3),
                    "signals": {
                        FEATURE_DISPLAY_NAMES.get(k, k): round(v, 3)
                        for k, v in agent_signals.items()
                    },
                }

            return {
                "feature_importance": importance[:20],
                "hourly_drivers": hourly_drivers,
                "model_type": "XGBoost + LightGBM + Ridge (stacked) + Agent Signal Adjustment",
                "total_features": n_features,
                "agent_signal_adjustment": agent_summary,
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
