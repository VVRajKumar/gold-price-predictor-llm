"""Data fetchers package."""

from src.data_fetchers.market_data import MarketDataFetcher
from src.data_fetchers.news_data import NewsDataFetcher
from src.data_fetchers.macro_data import MacroDataFetcher
from src.data_fetchers.etf_data import ETFDataFetcher

__all__ = [
    "MarketDataFetcher",
    "NewsDataFetcher",
    "MacroDataFetcher",
    "ETFDataFetcher",
]
