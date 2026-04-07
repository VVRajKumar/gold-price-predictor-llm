"""
Market data fetcher – pulls price/volume data from Yahoo Finance.
"""

import time
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger
from cachetools import TTLCache

from ..config import (
    GOLD_TICKER, SILVER_TICKER, OIL_TICKER, DXY_TICKER,
    USDINR_TICKER, NIFTY_TICKER, INDIA_VIX_TICKER,
    HISTORICAL_LOOKBACK_DAYS,
)

# Cache price data for 15 min to avoid hammering the API
_cache = TTLCache(maxsize=64, ttl=900)

# Retry settings for transient Yahoo Finance errors (rate limits, server errors)
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 2  # seconds


def _yf_download_with_retry(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    max_retries: int = _MAX_RETRIES,
) -> pd.DataFrame:
    """Wrapper around yf.download with exponential backoff for transient errors."""
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                interval=interval,
                progress=False,
            )
            if df is None:
                df = pd.DataFrame()
            if not df.empty:
                return df
            # Empty result on first attempt may be transient; retry
            if attempt < max_retries - 1:
                wait = _INITIAL_BACKOFF * (2 ** attempt)
                logger.info(f"Empty result for {ticker}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
        except Exception as e:
            err_str = str(e)
            is_transient = any(kw in err_str for kw in ("Rate", "Too Many", "NoneType", "timeout", "500", "503"))
            if is_transient and attempt < max_retries - 1:
                wait = _INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(f"Transient error for {ticker}: {e} — retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
            else:
                logger.error(f"Error fetching {ticker}: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


class MarketDataFetcher:
    """Fetch OHLCV market data for gold and correlated assets."""

    TICKERS = {
        "gold": GOLD_TICKER,
        "silver": SILVER_TICKER,
        "oil": OIL_TICKER,
        "usd_index": DXY_TICKER,
        "usdinr": USDINR_TICKER,
        "nifty50": NIFTY_TICKER,
        "india_vix": INDIA_VIX_TICKER,
    }

    # ------------------------------------------------------------------ #
    def fetch_ticker(
        self,
        ticker: str,
        period_days: int = HISTORICAL_LOOKBACK_DAYS,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Return OHLCV DataFrame for a single ticker."""
        cache_key = f"{ticker}_{period_days}_{interval}"
        if cache_key in _cache:
            return _cache[cache_key].copy()

        # Use UTC calendar dates so localhost and Streamlit Cloud fetch the same 1d candle window.
        # yfinance 'end' is exclusive, so add 1 day to include today's candles.
        end_date = datetime.utcnow().date() + timedelta(days=1)
        start_date = end_date - timedelta(days=period_days + 1)
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")

        try:
            df = _yf_download_with_retry(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
            )
            if df.empty:
                logger.warning(f"No data returned for {ticker}")
                return pd.DataFrame()

            # Flatten multi-level columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            # Ensure no duplicate columns – keep first occurrence
            df = df.loc[:, ~df.columns.duplicated()]

            if isinstance(df.index, pd.DatetimeIndex) and df.index.tz is not None:
                df.index = df.index.tz_convert(None)

            # Enforce a strict calendar-day window from the latest available candle.
            # This keeps chart ranges consistent even if the provider returns extra history.
            if period_days and isinstance(df.index, pd.DatetimeIndex) and not df.empty:
                end_ts = df.index.max()
                start_ts = end_ts - pd.Timedelta(days=period_days)
                df = df[df.index >= start_ts]
                df = df.sort_index()

            _cache[cache_key] = df
            return df.copy()
        except Exception as e:
            logger.error(f"Error fetching {ticker}: {e}")
            return pd.DataFrame()

    # ------------------------------------------------------------------ #
    def fetch_all(self, period_days: int = HISTORICAL_LOOKBACK_DAYS) -> dict[str, pd.DataFrame]:
        """Fetch data for every tracked ticker."""
        results = {}
        for name, ticker in self.TICKERS.items():
            results[name] = self.fetch_ticker(ticker, period_days)
        return results

    # ------------------------------------------------------------------ #
    def get_usdinr_rate(self) -> float:
        """Return latest USD/INR exchange rate, fallback to 85.5 if unavailable."""
        df = self.fetch_ticker(USDINR_TICKER, period_days=7)
        if not df.empty:
            close = df["Close"].squeeze()
            if hasattr(close, "iloc") and len(close) > 0:
                rate = float(close.iloc[-1])
                logger.info(f"USD/INR rate: {rate:.2f}")
                return rate
        logger.warning("USD/INR rate unavailable, using fallback 85.5")
        return 85.5  # updated fallback

    # ------------------------------------------------------------------ #
    def get_usdinr_series(self, period_days: int = 90, interval: str = "1d") -> pd.Series:
        """Return a time-indexed Series of USD/INR close rates.

        Used for time-aligned conversion of historical USD data to INR so each
        candle uses the FX rate from its own trading day rather than today's
        spot rate.
        """
        df = self.fetch_ticker(USDINR_TICKER, period_days=period_days, interval=interval)
        if df.empty:
            return pd.Series(dtype=float)
        close = df["Close"].squeeze()
        if isinstance(close, pd.DataFrame):
            close = close.iloc[:, 0]
        return pd.to_numeric(close, errors="coerce").dropna()

    # ------------------------------------------------------------------ #
    def convert_usd_to_inr(
        self,
        usd_df: pd.DataFrame,
        period_days: int = 90,
        interval: str = "1d",
    ) -> pd.DataFrame:
        """Convert OHLC columns from USD/oz to INR/10g using time-aligned FX rates.

        Always fetches **daily** FX rates (INR=X hourly is unreliable) and
        forward-fills them to match the gold data timestamps.
        Returns a NEW DataFrame — the input is never mutated.
        """
        result = usd_df.copy()
        oz_to_10g = 10.0 / 31.1035

        # Guard: if Close values are already in INR range (>10,000), skip conversion
        if "Close" in result.columns:
            median_close = pd.to_numeric(result["Close"], errors="coerce").median()
            if pd.notna(median_close) and median_close > 10_000:
                logger.warning(
                    f"Data appears already converted (median Close={median_close:.0f}), skipping"
                )
                return result

        # Always use daily FX series – most reliable granularity for INR=X
        fx = self.get_usdinr_series(period_days=period_days + 30, interval="1d")

        if fx.empty:
            spot = self.get_usdinr_rate()
            logger.warning(f"No FX series available; using spot {spot:.2f} for all rows")
            for col in ("Open", "High", "Low", "Close"):
                if col in result.columns:
                    result[col] = result[col] * spot * oz_to_10g
            return result

        # Normalise FX index to tz-naive for alignment
        if isinstance(fx.index, pd.DatetimeIndex) and fx.index.tz is not None:
            fx.index = fx.index.tz_convert(None)

        # Reindex daily FX to match gold data timestamps (may be hourly), forward-fill
        try:
            fx_aligned = fx.reindex(result.index, method="ffill")
            fx_aligned = fx_aligned.bfill()
            fx_aligned = fx_aligned.fillna(self.get_usdinr_rate())
        except Exception as e:
            # Fallback: if reindex fails (index type mismatch on some envs),
            # map each timestamp to the nearest daily FX rate manually.
            logger.warning(f"FX reindex failed ({e}), using date-based merge")
            spot = self.get_usdinr_rate()
            fx_map = dict(zip(fx.index.date if hasattr(fx.index, 'date') else fx.index, fx.values))
            rates = []
            for ts in result.index:
                d = ts.date() if hasattr(ts, 'date') else ts
                rates.append(fx_map.get(d, spot))
            fx_aligned = pd.Series(rates, index=result.index)

        for col in ("Open", "High", "Low", "Close"):
            if col in result.columns:
                result[col] = pd.to_numeric(result[col], errors="coerce").values * fx_aligned.values * oz_to_10g

        # Verify conversion actually produced INR-scale values
        if "Close" in result.columns:
            post_median = pd.to_numeric(result["Close"], errors="coerce").median()
            if pd.notna(post_median) and post_median < 10_000:
                logger.error(
                    f"FX conversion appears to have failed (post-median={post_median:.0f}),"
                    f" forcing spot-rate conversion"
                )
                spot = self.get_usdinr_rate()
                for col2 in ("Open", "High", "Low", "Close"):
                    if col2 in result.columns:
                        # Values are still in USD, apply full conversion
                        result[col2] = pd.to_numeric(result[col2], errors="coerce") * spot * oz_to_10g

        logger.info(
            f"FX conversion applied: rate range {fx_aligned.min():.2f}–{fx_aligned.max():.2f}"
        )
        return result

    # ------------------------------------------------------------------ #
    def get_gold_inr_price(self, period_days: int = 7) -> float:
        """Return gold price in INR per 10 grams (Indian standard unit)."""
        usd_rate = self.get_usdinr_rate()
        df = self.fetch_ticker(GOLD_TICKER, period_days=period_days)
        if df.empty:
            return 0.0
        close = df["Close"].squeeze()
        if hasattr(close, "iloc") and len(close) > 0:
            usd_per_oz = float(close.iloc[-1])
            # 1 troy oz = 31.1035 grams → per 10g
            inr_per_10g = (usd_per_oz / 31.1035) * 10 * usd_rate
            return round(inr_per_10g, 2)
        return 0.0

    # ------------------------------------------------------------------ #
    def get_gold_summary(self, period_days: int = 30) -> dict:
        """Return a compact dict summarising recent gold price action in INR/10g."""
        df = self.fetch_ticker(GOLD_TICKER, period_days)
        if df.empty:
            return {"error": "No gold data available"}

        close = pd.to_numeric(df["Close"].squeeze(), errors="coerce").dropna()
        high = pd.to_numeric(df["High"].squeeze(), errors="coerce").dropna()
        low = pd.to_numeric(df["Low"].squeeze(), errors="coerce").dropna()
        volume = pd.to_numeric(df["Volume"].squeeze(), errors="coerce").dropna() if "Volume" in df else pd.Series(dtype=float)

        if close.empty:
            logger.warning("Gold data returned but Close column has no valid values")
            return {"error": "No valid gold close prices available"}

        # Convert USD/oz to INR/10g using time-aligned FX rates
        fx = self.get_usdinr_series(period_days=period_days + 30)
        if fx.empty:
            factor = self.get_usdinr_rate() * 10.0 / 31.1035
            close_inr = close * factor
            high_inr = high * factor
            low_inr = low * factor
        else:
            oz_to_10g = 10.0 / 31.1035
            fx_aligned = fx.reindex(close.index, method="ffill").bfill().fillna(self.get_usdinr_rate())
            close_inr = close * fx_aligned * oz_to_10g
            high_inr = high * fx_aligned.reindex(high.index, method="ffill").bfill().fillna(self.get_usdinr_rate()) * oz_to_10g
            low_inr = low * fx_aligned.reindex(low.index, method="ffill").bfill().fillna(self.get_usdinr_rate()) * oz_to_10g

        current = float(close_inr.iloc[-1])
        prev = float(close_inr.iloc[-2]) if len(close_inr) > 1 else current
        daily_change_pct = 0.0 if prev == 0 else (current - prev) / prev * 100

        return {
            "current_price": round(current, 2),
            "prev_close": round(prev, 2),
            "daily_change_pct": round(float(daily_change_pct), 3),
            "30d_high": round(float(high_inr.max()), 2) if not high_inr.empty else round(current, 2),
            "30d_low": round(float(low_inr.min()), 2) if not low_inr.empty else round(current, 2),
            "30d_avg": round(float(close_inr.mean()), 2),
            "30d_volatility": round(float(close_inr.pct_change().dropna().std() * 100), 4) if len(close_inr) > 2 else 0.0,
            "volume_latest": int(volume.iloc[-1]) if not volume.empty else 0,
            "data_points": int(len(close_inr)),
            "as_of": close.index[-1].strftime("%Y-%m-%d"),
            "currency": "INR",
            "unit": "per 10 grams",
        }

    # ------------------------------------------------------------------ #
    def get_correlation_snapshot(self, period_days: int = 90) -> dict:
        """Compute recent correlations between gold and other assets."""
        all_data = self.fetch_all(period_days)
        gold_close = all_data.get("gold")
        if gold_close is None or gold_close.empty:
            return {}

        gold_returns = gold_close["Close"].squeeze().pct_change().dropna()
        correlations = {}
        for name, df in all_data.items():
            if name == "gold" or df.empty:
                continue
            other_returns = df["Close"].squeeze().pct_change().dropna()
            # Align indices
            common = gold_returns.index.intersection(other_returns.index)
            if len(common) < 10:
                continue
            corr = gold_returns.loc[common].corr(other_returns.loc[common])
            correlations[name] = round(float(corr), 4)

        return correlations
