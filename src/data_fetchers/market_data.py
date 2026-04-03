"""
Market data fetcher – pulls price/volume data from Yahoo Finance.
"""

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
        end_date = datetime.utcnow().date()
        start_date = end_date - timedelta(days=period_days)
        logger.info(f"Fetching {ticker} from {start_date} to {end_date}")

        try:
            df = yf.download(
                ticker,
                start=start_date.strftime("%Y-%m-%d"),
                end=end_date.strftime("%Y-%m-%d"),
                interval=interval,
                progress=False,
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
            return df
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
        """Return latest USD/INR exchange rate, fallback to 83.5 if unavailable."""
        df = self.fetch_ticker(USDINR_TICKER, period_days=7)
        if not df.empty:
            close = df["Close"].squeeze()
            if hasattr(close, "iloc") and len(close) > 0:
                return float(close.iloc[-1])
        return 83.5  # reasonable fallback

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

        # Convert USD/oz to INR/10g
        usdinr = self.get_usdinr_rate()
        factor = usdinr * 10.0 / 31.1035  # USD/oz -> INR/10g
        close_inr = close * factor
        high_inr = high * factor
        low_inr = low * factor

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
