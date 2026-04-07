"""
Macroeconomic data fetcher – FRED API for interest rates, CPI, money supply, etc.
"""

import time
from datetime import datetime, timedelta

import pandas as pd
from loguru import logger
from cachetools import TTLCache

from ..config import FRED_API_KEY, FRED_SERIES

_cache = TTLCache(maxsize=32, ttl=3600)  # 1-hour cache

# Retry settings for transient FRED API errors
_MAX_RETRIES = 3
_INITIAL_BACKOFF = 2  # seconds


class MacroDataFetcher:
    """Fetch macroeconomic indicators from FRED."""

    def __init__(self):
        self._api_key = FRED_API_KEY

    _SERIES_FALLBACKS = {
        # Reserved for series-specific fallback handling.
    }

    # ------------------------------------------------------------------ #
    def _fred_request(self, series_id: str, lookback_days: int = 365) -> pd.DataFrame:
        """Low-level FRED API call with retry logic for transient errors."""
        cache_key = f"fred_{series_id}_{lookback_days}"
        if cache_key in _cache:
            return _cache[cache_key]

        if not self._api_key:
            logger.warning("FRED_API_KEY not set – returning empty frame")
            return pd.DataFrame()

        import requests

        start = (datetime.now() - timedelta(days=lookback_days)).strftime("%Y-%m-%d")
        url = "https://api.stlouisfed.org/fred/series/observations"
        params = {
            "series_id": series_id,
            "api_key": self._api_key,
            "file_type": "json",
            "observation_start": start,
            "sort_order": "desc",
        }

        for attempt in range(_MAX_RETRIES):
            try:
                resp = requests.get(url, params=params, timeout=15)
                resp.raise_for_status()
                data = resp.json().get("observations", [])
                if not data:
                    return pd.DataFrame()

                df = pd.DataFrame(data)
                df["date"] = pd.to_datetime(df["date"])
                df["value"] = pd.to_numeric(df["value"], errors="coerce")
                df = df.dropna(subset=["value"]).set_index("date")[["value"]]
                df = df.sort_index()
                _cache[cache_key] = df
                return df
            except Exception as e:
                msg = str(e)
                is_transient = any(kw in msg for kw in ("500", "502", "503", "timeout", "Timeout", "ConnectionError"))
                if "400" in msg:
                    logger.warning(f"FRED 400 for {series_id}: {msg}")
                    return pd.DataFrame()
                elif is_transient and attempt < _MAX_RETRIES - 1:
                    wait = _INITIAL_BACKOFF * (2 ** attempt)
                    logger.warning(f"FRED transient error for {series_id}: {msg} — retrying in {wait}s")
                    time.sleep(wait)
                else:
                    logger.error(f"FRED error for {series_id}: {msg}")
                    return pd.DataFrame()
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    def fetch_series(self, name: str, lookback_days: int = 365) -> pd.DataFrame:
        """Fetch a named macro series from FRED."""
        series_id = FRED_SERIES.get(name)
        if not series_id:
            logger.warning(f"Unknown series name: {name}")
            return pd.DataFrame()

        candidates = self._SERIES_FALLBACKS.get(name, [series_id])
        for sid in candidates:
            df = self._fred_request(sid, lookback_days)
            if not df.empty:
                return df

        logger.warning(f"No FRED data returned for series '{name}' using candidates: {candidates}")
        return pd.DataFrame()

    # ------------------------------------------------------------------ #
    def fetch_all(self, lookback_days: int = 365) -> dict[str, pd.DataFrame]:
        """Fetch all configured FRED series."""
        results = {}
        for name in FRED_SERIES:
            results[name] = self.fetch_series(name, lookback_days)
        return results

    # ------------------------------------------------------------------ #
    def get_macro_summary(self) -> dict:
        """Return a compact summary of key macro indicators."""
        summary = {}
        for name in FRED_SERIES:
            df = self.fetch_series(name, lookback_days=180)
            if df.empty:
                summary[name] = {"latest": None, "change_6m": None}
                continue

            latest_val = float(df["value"].iloc[-1])
            first_val = float(df["value"].iloc[0])
            change = round(latest_val - first_val, 4)

            summary[name] = {
                "latest": round(latest_val, 4),
                "date": df.index[-1].strftime("%Y-%m-%d"),
                "6m_change": change,
                "6m_change_pct": round(change / abs(first_val) * 100, 2) if first_val != 0 else 0,
            }
        return summary
