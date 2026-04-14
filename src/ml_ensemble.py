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

Data sources:
  - Training/prediction uses COMEX (GC=F) hourly candles because MCX (GOLD.NS)
    hourly data is unreliable on Yahoo Finance (only daily candles available).
  - All prices are internally USD/oz.  Conversion to INR/10g happens once at the
    end using time-aligned USD/INR rates.
  - MCX is used as the primary source for daily prices in agents, accuracy
    tracking, and UI display (see market_data.py).
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
from .data_fetchers.market_data import MarketDataFetcher, _safe_col
from .guardrails import validate_xgb_predictions, validate_price_series
from .residual_learner import ResidualLearner

# ── Feature engineering constants ────────────────────────────────────
_LAGS = [1, 2, 3, 6, 12, 24]
_ROLL_WINDOWS = [6, 12, 24]
_MIN_HISTORY = 25            # minimum samples for feature construction
_TRAIN_PERIOD_DAYS = 90      # 3 months of hourly data (~2160 bars)
_MIN_TRAIN_SAMPLES = 200     # hard floor after cleanup
_DEFAULT_INR_PRICE = 92_000  # fallback gold price (INR/10g) when market data unavailable

# ── Session-aware prediction constants ──────────────────────────────
# Total deviation cap: max cumulative movement from starting price (±5%)
_MAX_TOTAL_DEV_PCT = 0.05
# Per-step cap: max single-hour move (±1.0%)
_MAX_STEP_PCT = 0.01
# Band floor: minimum band half-width (fraction of price)
_BAND_BASE_FLOOR = 0.003
# Band per-horizon floor: scales with √(h+1)
_BAND_HORIZON_FLOOR = 0.005
# Number of recent days to compute intra-day session patterns
_SESSION_PATTERN_DAYS = 14
# Minimum number of data points per hour-of-day bin to use session patterns
_MIN_SESSION_SAMPLES = 5
# Mean-reversion strength: only for extreme deviations (>3% from start)
_MEAN_REVERSION_THRESHOLD_PCT = 0.03
_MEAN_REVERSION_STRENGTH = 0.15
# Target session-pattern per-hour move (fraction of price, 0.05%).
# Raw COMEX hourly returns are ~0.01-0.03%, which is barely visible on charts.
# This target scales them up to ~0.05% per hour for meaningful chart movement
# while staying within realistic gold volatility.
_TARGET_SESSION_MOVE = 0.0005
# Maximum session-pattern amplification (prevents over-scaling if raw returns
# are extremely small, e.g. during very quiet markets)
_MAX_SESSION_SCALE = 10.0

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
    learned_weight_adjustments: Optional[dict[str, float]] = None,
) -> float:
    """Compute a directional adjustment factor from agent signals.

    Returns a multiplier in the range ``1 - MAX_ADJ … 1 + MAX_ADJ``
    (e.g. 0.985 … 1.015 for ±1.5 %).

    The composite signal is a weighted average of the 8 agent scores,
    each in [-1, +1], mapped to a bounded percentage adjustment.

    If ``learned_weight_adjustments`` is provided (from ResidualLearner),
    agent weights are dynamically multiplied by learned accuracy factors.
    """
    if not agent_signals:
        return 1.0

    weighted_sum = 0.0
    total_weight = 0.0
    for name, base_weight in _AGENT_WEIGHTS.items():
        val = agent_signals.get(name, 0.0)
        # Apply learned weight adjustment if available
        learned_mult = learned_weight_adjustments.get(name, 1.0) if learned_weight_adjustments else 1.0
        effective_weight = base_weight * learned_mult
        weighted_sum += val * effective_weight
        total_weight += effective_weight

    if total_weight == 0:
        return 1.0

    # Composite signal in [-1, +1]
    composite = max(-1.0, min(1.0, weighted_sum / total_weight))

    # Map to a bounded percentage adjustment
    adjustment_pct = composite * _MAX_AGENT_ADJUSTMENT_PCT / 100.0
    return 1.0 + adjustment_pct


def _extract_session_pattern(close_series: pd.Series) -> dict[int, float]:
    """Extract zero-centered hourly return pattern by hour-of-day from recent data.

    The returned pattern has its mean subtracted so it represents SHAPE only
    (which hours are relatively more bullish vs bearish), not overall direction.
    The ML model and agent signals provide the directional bias separately.

    Returns a dict mapping hour (0-23) to mean-subtracted hourly return (fraction).
    If insufficient data, returns an empty dict.
    """
    if len(close_series) < 48:  # need at least 2 days of hourly data
        return {}

    returns = close_series.pct_change().dropna()
    if returns.empty:
        return {}

    # Group returns by hour-of-day (skip entries without valid timestamps)
    hour_returns: dict[int, list[float]] = {}
    for ts, ret in returns.items():
        if not hasattr(ts, 'hour'):
            continue  # skip entries without valid timestamp
        hour_returns.setdefault(ts.hour, []).append(float(ret))

    # Compute average return per hour, only if we have enough samples
    raw_pattern: dict[int, float] = {}
    for h in range(24):
        rets = hour_returns.get(h, [])
        if len(rets) >= _MIN_SESSION_SAMPLES:
            # Use trimmed mean (exclude top/bottom 10%) to reduce outlier impact
            sorted_rets = sorted(rets)
            trim = max(1, len(sorted_rets) // 10)
            trimmed = sorted_rets[trim:-trim] if len(sorted_rets) > 2 * trim else sorted_rets
            raw_pattern[h] = float(np.mean(trimmed))
        else:
            raw_pattern[h] = 0.0

    if not raw_pattern:
        return {}

    # Zero-center: subtract mean so pattern represents SHAPE only
    # (relative bullish/bearish hours), not overall market direction
    mean_ret = np.mean(list(raw_pattern.values()))
    pattern = {h: r - mean_ret for h, r in raw_pattern.items()}

    if pattern:
        bullish_hours = sorted([h for h, r in pattern.items() if r > 0.00005])
        bearish_hours = sorted([h for h, r in pattern.items() if r < -0.00005])
        avg_abs = np.mean([abs(r) for r in pattern.values()]) * 100
        logger.info(
            f"Session pattern (zero-centered): {len(pattern)} hours, "
            f"bullish_hours={bullish_hours}, bearish_hours={bearish_hours}, "
            f"avg |return|={avg_abs:.4f}%"
        )

    return pattern


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
        # Using COMEX (GC=F) because MCX (GOLD.NS) hourly data is unavailable
        # on Yahoo Finance.  COMEX closely tracks MCX; the final predictions
        # are converted to INR/10g using time-aligned USD/INR rates.
        df = self._market.fetch_ticker("GC=F", period_days=_TRAIN_PERIOD_DAYS, interval="1h")
        if df.empty or "Close" not in df:
            logger.warning("ML ensemble: no training data")
            return False

        close = pd.to_numeric(_safe_col(df, "Close"), errors="coerce").dropna()
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

        Uses **session-pattern shaping**: combines the ML model's directional
        signal with actual intra-day return patterns extracted from recent
        COMEX history.  This produces realistic 24-hour shapes (hill, zigzag,
        flat, etc.) because the pattern reflects how gold *actually* behaves
        at each hour of the day.

        Architecture:
          1. Get ML model's 1-step-ahead prediction → directional bias
          2. Extract real per-hour-of-day return pattern from last 14 days
          3. Shape the 24h forecast: session pattern × directional bias × agents
          4. Apply safety caps (±1% per step, ±5% total)
          5. Compute quantile bands with horizon widening

        Why session-pattern shaping?
          - Autoregressive: tree models compound → monotonic curves
          - Direct multi-step: only 4 time features change → artificial patterns
          - Session pattern: uses ACTUAL historical return patterns → natural flow

        Returns list of dicts with keys:
            date, xgb_price, xgb_low, xgb_high,
            component_xgb, component_lgb, component_ridge
        """
        if not self._is_trained:
            logger.warning("ML ensemble: not trained, cannot predict")
            return []

        try:
            df = self._market.fetch_ticker("GC=F", period_days=30, interval="1h")
            if df.empty:
                return []

            close = pd.to_numeric(_safe_col(df, "Close"), errors="coerce").dropna()
            if len(close) < _MIN_HISTORY:
                return []

            values = close.to_numpy(dtype=float)
            actual_history = values.tolist()
            last_ts = close.index.max()

            usdinr = self._market.get_usdinr_rate()
            oz_to_10g = 10.0 / 31.1035

            ref_usd_price = float(values[-1])

            # ── 1. ML model's directional signal ──
            # Use the model's 1-step-ahead prediction to determine direction + magnitude.
            # This is the most reliable prediction from tree models.
            ts_1 = pd.Timestamp(last_ts) + pd.Timedelta(hours=1)
            feat_1 = _build_feature_vector(actual_history, ts_1).reshape(1, -1)

            p_xgb_1 = float(self._xgb_model.predict(feat_1)[0])
            p_lgb_1 = float(self._lgb_model.predict(
                pd.DataFrame(feat_1, columns=FEATURE_NAMES))[0])
            p_ridge_1 = float(self._ridge_model.predict(feat_1)[0])
            stack_1 = np.array([[p_xgb_1, p_lgb_1, p_ridge_1]])
            ml_direction_usd = float(self._meta_model.predict(stack_1)[0])

            # ML model's 1-step return (direction + magnitude)
            ml_1step_return = (ml_direction_usd - ref_usd_price) / ref_usd_price if ref_usd_price > 0 else 0.0
            ml_1step_return = max(-_MAX_STEP_PCT, min(_MAX_STEP_PCT, ml_1step_return))

            # ── 2. Agent adjustment (with learned weight tuning) ──
            learned_agent_weights = self._residual_learner.get_agent_weight_adjustments()
            agent_multiplier = _compute_agent_adjustment(agent_signals, learned_agent_weights)
            agent_adj_pct = (agent_multiplier - 1.0) * 100
            logger.info(
                f"Agent signal adjustment: {agent_adj_pct:+.3f}% "
                f"(multiplier={agent_multiplier:.5f})"
            )

            # ── 3. Extract real session pattern from recent COMEX data ──
            session_pattern = _extract_session_pattern(close)

            # ── 4. Build 24-hour price path using session pattern ──
            # Strategy: session pattern provides SHAPE (which hours are relatively
            # up vs down), while ML direction + agents provide overall DRIFT.
            # The session pattern is zero-centered, so without ML/agent bias the
            # prediction would oscillate around the starting price.
            preds: list[dict] = []
            X_for_shap: list[np.ndarray] = []
            prev_usd = ref_usd_price

            # Directional drift: ML's predicted return spread over 24 hours
            # plus agent signal spread over 24 hours.
            # This is a GENTLE per-hour drift that doesn't overpower session pattern.
            ml_drift_per_hour = ml_1step_return / PREDICTION_HOURS
            agent_drift_per_hour = (agent_multiplier - 1.0) / PREDICTION_HOURS

            for h in range(PREDICTION_HOURS):
                ts = pd.Timestamp(last_ts) + pd.Timedelta(hours=h + 1)

                # Build features for SHAP and component display
                feat = _build_feature_vector(actual_history, ts).reshape(1, -1)
                X_for_shap.append(feat.flatten())

                # Component predictions for UI transparency
                p_xgb = float(self._xgb_model.predict(feat)[0])
                p_lgb = float(self._lgb_model.predict(
                    pd.DataFrame(feat, columns=FEATURE_NAMES))[0])
                p_ridge = float(self._ridge_model.predict(feat)[0])

                # ── Session pattern: SHAPE component (zero-centered) ──
                # This determines whether this hour tends to be up or down
                # relative to the mean — the core of realistic intra-day flow
                target_hour = ts.hour
                session_return = session_pattern.get(target_hour, 0.0)

                # Scale session pattern to be visible on charts.
                # Raw returns are often ~0.01-0.05% per hour; scale to ~0.05-0.15%
                # for meaningful chart movement while keeping it realistic.
                avg_abs_return = (
                    np.mean([abs(v) for v in session_pattern.values()])
                    if session_pattern
                    else _TARGET_SESSION_MOVE
                )
                session_scale = max(1.0, _TARGET_SESSION_MOVE / max(avg_abs_return, 1e-8))
                session_scale = min(session_scale, _MAX_SESSION_SCALE)
                session_component = session_return * session_scale

                # ── Directional drift: ML + agents (applied evenly) ──
                # ML drift decays with horizon (near-term is more accurate)
                ml_decay = max(0.3, 1.0 - 0.03 * h)
                drift_component = ml_drift_per_hour * ml_decay + agent_drift_per_hour

                # ── Combined per-hour return ──
                combined_return = session_component + drift_component

                # ── Light mean-reversion for extreme deviations ──
                if ref_usd_price > 0:
                    cum_dev = (prev_usd - ref_usd_price) / ref_usd_price
                    if abs(cum_dev) > _MEAN_REVERSION_THRESHOLD_PCT:
                        revert = -cum_dev * _MEAN_REVERSION_STRENGTH
                        combined_return += revert

                # Apply return to get new price
                p_final_usd = prev_usd * (1.0 + combined_return)

                # ── Per-step cap: ±1.0% ──
                if prev_usd > 0:
                    max_step = prev_usd * _MAX_STEP_PCT
                    p_final_usd = max(prev_usd - max_step,
                                      min(p_final_usd, prev_usd + max_step))

                # ── Total deviation cap: ±5% ──
                if ref_usd_price > 0:
                    max_dev = ref_usd_price * _MAX_TOTAL_DEV_PCT
                    p_final_usd = max(ref_usd_price - max_dev,
                                      min(p_final_usd, ref_usd_price + max_dev))

                # ── Quantile bands ──
                lo_usd = float(self._xgb_lo.predict(feat)[0])
                hi_usd = float(self._xgb_hi.predict(feat)[0])
                band_half = (hi_usd - lo_usd) / 2.0
                widen_factor = math.sqrt(h + 1)
                band_half *= widen_factor
                min_band_half = ref_usd_price * _BAND_HORIZON_FLOOR * widen_factor
                band_half = max(band_half, min_band_half)

                lo_usd = p_final_usd - band_half
                hi_usd = p_final_usd + band_half
                lo_usd, hi_usd = min(lo_usd, hi_usd), max(lo_usd, hi_usd)

                # Convert to INR/10g
                p_final_inr = p_final_usd * usdinr * oz_to_10g
                lo_inr = lo_usd * usdinr * oz_to_10g
                hi_inr = hi_usd * usdinr * oz_to_10g
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
                    "component_xgb": round(p_xgb * usdinr * oz_to_10g, 2),
                    "component_lgb": round(p_lgb * usdinr * oz_to_10g, 2),
                    "component_ridge": round(p_ridge * usdinr * oz_to_10g, 2),
                })

                prev_usd = p_final_usd

            # Validate predictions
            current_inr = self._market.get_gold_inr_price()
            if not (isinstance(current_inr, (int, float)) and np.isfinite(current_inr) and current_inr > 0):
                current_inr = preds[0]["xgb_price"] if preds else _DEFAULT_INR_PRICE
            preds = validate_xgb_predictions(preds, current_inr)

            # Apply residual corrections from past errors.
            # Pass the current IST slot so slot-specific bias is applied.
            from .time_utils import current_slot_ist
            try:
                gen_slot = current_slot_ist().hour
            except Exception:
                gen_slot = None
            preds = self._residual_learner.correct_predictions(preds, generation_slot=gen_slot)

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
