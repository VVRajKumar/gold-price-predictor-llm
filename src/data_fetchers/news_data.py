"""
News data fetcher – pulls gold-related news from NewsAPI and RSS feeds.
"""

from datetime import datetime, timedelta
from typing import Optional

import feedparser
import requests
from loguru import logger
from cachetools import TTLCache

from ..config import NEWS_API_KEY

_cache = TTLCache(maxsize=16, ttl=1800)  # 30-min cache

# RSS feeds for gold / macro news
RSS_FEEDS = {
    "kitco_gold": "https://www.kitco.com/feed/gold-news",
    "reuters_commodities": "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best",
    "investing_gold": "https://www.investing.com/rss/news_14.rss",
}


class NewsDataFetcher:
    """Aggregate gold-relevant news from multiple sources."""

    # ------------------------------------------------------------------ #
    def fetch_newsapi(
        self,
        query: str = "gold price OR gold market OR gold futures",
        days_back: int = 3,
        page_size: int = 20,
    ) -> list[dict]:
        """Fetch articles from NewsAPI."""
        if not NEWS_API_KEY:
            logger.warning("NEWS_API_KEY not set – skipping newsapi")
            return []

        cache_key = f"newsapi_{query}_{days_back}"
        if cache_key in _cache:
            return _cache[cache_key]

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "pageSize": page_size,
            "language": "en",
            "apiKey": NEWS_API_KEY,
        }

        try:
            resp = requests.get(url, params=params, timeout=15)
            resp.raise_for_status()
            articles = resp.json().get("articles", [])
            cleaned = [
                {
                    "title": a.get("title", ""),
                    "description": a.get("description", ""),
                    "source": a.get("source", {}).get("name", ""),
                    "url": a.get("url", ""),
                    "published": a.get("publishedAt", ""),
                }
                for a in articles
                if a.get("title")
            ]
            _cache[cache_key] = cleaned
            return cleaned
        except Exception as e:
            logger.error(f"NewsAPI error: {e}")
            return []

    # ------------------------------------------------------------------ #
    def fetch_rss(self) -> list[dict]:
        """Fetch articles from curated RSS feeds."""
        cache_key = "rss_all"
        if cache_key in _cache:
            return _cache[cache_key]

        articles: list[dict] = []
        for name, url in RSS_FEEDS.items():
            try:
                feed = feedparser.parse(url)
                for entry in feed.entries[:10]:
                    articles.append(
                        {
                            "title": entry.get("title", ""),
                            "description": entry.get("summary", "")[:500],
                            "source": name,
                            "url": entry.get("link", ""),
                            "published": entry.get("published", ""),
                        }
                    )
            except Exception as e:
                logger.warning(f"RSS feed {name} failed: {e}")

        _cache[cache_key] = articles
        return articles

    # ------------------------------------------------------------------ #
    def fetch_geopolitics_news(self, days_back: int = 3) -> list[dict]:
        """Fetch geopolitics news that may affect gold."""
        queries = [
            "geopolitics war conflict sanctions",
            "central bank gold reserves",
            "federal reserve interest rate decision",
            "inflation recession economy",
            "BRICS dollar dedollarization",
        ]
        all_articles: list[dict] = []
        for q in queries:
            all_articles.extend(self.fetch_newsapi(q, days_back, page_size=5))
        return all_articles

    # ------------------------------------------------------------------ #
    def get_all_news(self, days_back: int = 3) -> dict[str, list[dict]]:
        """Return all news grouped by category."""
        return {
            "gold_market": self.fetch_newsapi(days_back=days_back),
            "geopolitics": self.fetch_geopolitics_news(days_back),
            "rss_feeds": self.fetch_rss(),
        }
