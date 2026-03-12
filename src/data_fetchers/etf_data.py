"""
ETF & mining-stock data fetcher.
"""

from datetime import datetime, timedelta

import pandas as pd
import yfinance as yf
from loguru import logger
from cachetools import TTLCache

from src.config import GOLD_ETF_TICKERS, GOLD_MINER_TICKERS, HISTORICAL_LOOKBACK_DAYS

_cache = TTLCache(maxsize=64, ttl=900)


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
                df = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    progress=False,
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
                df = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=end.strftime("%Y-%m-%d"),
                    progress=False,
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
            if df.empty or len(df) < 2:
                continue

            close = df["Close"].squeeze()
            volume = df["Volume"].squeeze() if "Volume" in df else pd.Series(dtype=float)

            avg_recent_vol = float(volume.tail(5).mean()) if len(volume) > 0 else 0
            avg_older_vol = float(volume.tail(20).head(15).mean()) if len(volume) > 15 else 0

            price_change = (
                (float(close.iloc[-1]) - float(close.iloc[0]))
                / float(close.iloc[0])
                * 100
            )

            summary[ticker] = {
                "current_price": round(float(close.iloc[-1]), 2),
                "period_return_pct": round(price_change, 2),
                "avg_volume_5d": int(avg_recent_vol),
                "avg_volume_15d": int(avg_older_vol),
                "volume_trend": (
                    "increasing" if avg_recent_vol > avg_older_vol * 1.1
                    else "decreasing" if avg_recent_vol < avg_older_vol * 0.9
                    else "stable"
                ),
            }

        return summary
