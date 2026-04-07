"""
ETF & mining-stock data fetcher.
"""

import time
from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger
from cachetools import TTLCache

from ..config import GOLD_ETF_TICKERS, GOLD_MINER_TICKERS, GOLD_FUND_TICKERS, HISTORICAL_LOOKBACK_DAYS

_cache = TTLCache(maxsize=64, ttl=900)

# Retry settings for transient Yahoo Finance errors
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 2  # seconds


def _yf_download_safe(
    ticker: str,
    start: str,
    end: str,
    max_retries: int = _MAX_RETRIES,
) -> pd.DataFrame:
    """Wrapper around yf.download with retry logic for transient errors."""
    for attempt in range(max_retries):
        try:
            df = yf.download(
                ticker,
                start=start,
                end=end,
                progress=False,
            )
            if df is None:
                df = pd.DataFrame()
            if not df.empty:
                # Drop rows where all columns are NaN (delisted tickers
                # can return structurally non-empty but all-NaN DataFrames).
                df = df.dropna(how="all")
                if not df.empty:
                    return df
            if attempt < max_retries - 1:
                wait = _INITIAL_BACKOFF * (2 ** attempt)
                logger.info(f"Empty result for {ticker}, retrying in {wait}s (attempt {attempt + 1}/{max_retries})")
                time.sleep(wait)
        except Exception as e:
            err_str = str(e)
            is_transient = any(kw in err_str for kw in (
                "Rate", "Too Many", "NoneType", "timeout", "500", "503",
                "No objects to concatenate", "dictionary changed size",
            ))
            if is_transient and attempt < max_retries - 1:
                wait = _INITIAL_BACKOFF * (2 ** attempt)
                logger.warning(f"Transient error for {ticker}: {e} — retrying in {wait}s")
                time.sleep(wait)
            else:
                logger.warning(f"ETF {ticker} fetch error: {e}")
                return pd.DataFrame()
    return pd.DataFrame()


def _safe_float(val, default: float = 0.0) -> float:
    """Convert value to float, returning default if NaN or conversion fails."""
    try:
        v = float(val)
        return default if pd.isna(v) else v
    except (TypeError, ValueError):
        return default


def _safe_int(val, default: int = 0) -> int:
    """Convert value to int, returning default if NaN or conversion fails."""
    try:
        v = float(val)
        return default if pd.isna(v) else int(v)
    except (TypeError, ValueError):
        return default


class ETFDataFetcher:
    """Fetch Gold ETF and miner data."""

    # ------------------------------------------------------------------ #
    def fetch_etf_prices(
        self, period_days: int = HISTORICAL_LOOKBACK_DAYS
    ) -> dict[str, pd.DataFrame]:
        """Download price history for major gold ETFs."""
        cache_key = f"etfs_{period_days}"
        if cache_key in _cache:
            return _cache[cache_key]

        end = datetime.now()
        start = end - timedelta(days=period_days)
        results = {}

        for ticker in GOLD_ETF_TICKERS:
            try:
                df = _yf_download_safe(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df = df.loc[:, ~df.columns.duplicated()]
                results[ticker] = df
            except Exception as e:
                logger.warning(f"ETF {ticker} fetch error: {e}")

        _cache[cache_key] = results
        return results

    # ------------------------------------------------------------------ #
    def fetch_miner_prices(
        self, period_days: int = HISTORICAL_LOOKBACK_DAYS
    ) -> dict[str, pd.DataFrame]:
        """Download price history for gold miners."""
        cache_key = f"miners_{period_days}"
        if cache_key in _cache:
            return _cache[cache_key]

        end = datetime.now()
        start = end - timedelta(days=period_days)
        results = {}

        for ticker in GOLD_MINER_TICKERS:
            try:
                df = _yf_download_safe(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df = df.loc[:, ~df.columns.duplicated()]
                results[ticker] = df
            except Exception as e:
                logger.warning(f"Miner {ticker} fetch error: {e}")

        _cache[cache_key] = results
        return results

    # ------------------------------------------------------------------ #
    def get_etf_flow_summary(self, period_days: int = 30) -> dict:
        """Estimate ETF fund flows from volume and price changes."""
        etf_data = self.fetch_etf_prices(period_days)
        summary = {}

        for ticker, df in etf_data.items():
            try:
                if df.empty or len(df) < 2:
                    continue

                close = df["Close"].squeeze()
                volume = df["Volume"].squeeze() if "Volume" in df else pd.Series(dtype=float)

                avg_recent_vol = _safe_float(volume.tail(5).mean()) if len(volume) > 0 else 0.0
                avg_older_vol = _safe_float(volume.tail(20).head(15).mean()) if len(volume) > 15 else 0.0

                close_last = _safe_float(close.iloc[-1])
                close_first = _safe_float(close.iloc[0])

                if close_first == 0:
                    logger.warning(f"ETF {ticker}: zero initial price, skipping summary")
                    continue

                price_change = (close_last - close_first) / close_first * 100

                summary[ticker] = {
                    "current_price": round(close_last, 2),
                    "period_return_pct": round(price_change, 2),
                    "avg_volume_5d": _safe_int(avg_recent_vol),
                    "avg_volume_15d": _safe_int(avg_older_vol),
                    "volume_trend": (
                        "increasing" if avg_recent_vol > avg_older_vol * 1.1
                        else "decreasing" if avg_recent_vol < avg_older_vol * 0.9
                        else "stable"
                    ),
                }
            except Exception as e:
                logger.warning(f"ETF {ticker}: skipping summary due to {type(e).__name__}: {e}")

        return summary

    # ------------------------------------------------------------------ #
    def fetch_fund_prices(
        self, period_days: int = HISTORICAL_LOOKBACK_DAYS
    ) -> dict[str, pd.DataFrame]:
        """Download price history for Indian gold mutual funds."""
        cache_key = f"funds_{period_days}"
        if cache_key in _cache:
            return _cache[cache_key]

        end = datetime.now()
        start = end - timedelta(days=period_days)
        results = {}

        for ticker in GOLD_FUND_TICKERS:
            try:
                df = _yf_download_safe(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                )
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.droplevel(1)
                df = df.loc[:, ~df.columns.duplicated()]
                results[ticker] = df
            except Exception as e:
                logger.warning(f"Fund {ticker} fetch error: {e}")

        _cache[cache_key] = results
        return results

    # ------------------------------------------------------------------ #
    def get_fund_summary(self, period_days: int = 30) -> dict:
        """Return NAV-based summary for gold mutual funds."""
        fund_data = self.fetch_fund_prices(period_days)
        summary = {}

        for ticker, df in fund_data.items():
            try:
                if df.empty or len(df) < 2:
                    continue

                close = df["Close"].squeeze()
                close_last = _safe_float(close.iloc[-1])
                close_first = _safe_float(close.iloc[0])

                if close_first == 0:
                    logger.warning(f"Fund {ticker}: zero initial price, skipping summary")
                    continue

                price_change = (close_last - close_first) / close_first * 100

                summary[ticker] = {
                    "current_nav": round(close_last, 2),
                    "period_return_pct": round(price_change, 2),
                }
            except Exception as e:
                logger.warning(f"Fund {ticker}: skipping summary due to {type(e).__name__}: {e}")

        return summary
