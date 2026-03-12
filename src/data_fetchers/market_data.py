"""
Market data fetcher – pulls price/volume data from Yahoo Finance.
"""

from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
import yfinance as yf
from loguru import logger
from cachetools import TTLCache

from src.config import (
    GOLD_TICKER, SILVER_TICKER, OIL_TICKER, DXY_TICKER,
    SP500_TICKER, TREASURY_10Y, VIX_TICKER,
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
        "sp500": SP500_TICKER,
        "treasury_10y": TREASURY_10Y,
        "vix": VIX_TICKER,
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
            return _cache[cache_key]

        end = datetime.now()
        start = end - timedelta(days=period_days)
        logger.info(f"Fetching {ticker} from {start.date()} to {end.date()}")

        try:
            df = yf.download(
                ticker,
                start=start.strftime("%Y-%m-%d"),
                end=end.strftime("%Y-%m-%d"),
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
    def get_gold_summary(self, period_days: int = 30) -> dict:
        """Return a compact dict summarising recent gold price action."""
        df = self.fetch_ticker(GOLD_TICKER, period_days)
        if df.empty:
            return {"error": "No gold data available"}

        latest = df.iloc[-1]
        prev = df.iloc[-2] if len(df) > 1 else latest
        return {
            "current_price": round(float(df["Close"].squeeze().iloc[-1]), 2),
            "prev_close": round(float(df["Close"].squeeze().iloc[-2] if len(df) > 1 else df["Close"].squeeze().iloc[-1]), 2),
            "daily_change_pct": round(
                (float(df["Close"].squeeze().iloc[-1]) - float(df["Close"].squeeze().iloc[-2 if len(df) > 1 else -1]))
                / float(df["Close"].squeeze().iloc[-2 if len(df) > 1 else -1]) * 100, 3
            ),
            "30d_high": round(float(df["High"].squeeze().max()), 2),
            "30d_low": round(float(df["Low"].squeeze().min()), 2),
            "30d_avg": round(float(df["Close"].squeeze().mean()), 2),
            "30d_volatility": round(float(df["Close"].squeeze().pct_change().std() * 100), 4),
            "volume_latest": int(df["Volume"].squeeze().iloc[-1]) if "Volume" in df else 0,
            "data_points": len(df),
            "as_of": df.index[-1].strftime("%Y-%m-%d"),
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
