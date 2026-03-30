"""
News data fetcher – pulls gold-related news from NewsAPI and RSS feeds.
"""

from datetime import datetime, timedelta
import threading
from typing import Optional
import re

import feedparser
import requests
from loguru import logger
from cachetools import TTLCache

from ..config import NEWS_API_KEY

_cache = TTLCache(maxsize=16, ttl=1800)  # 30-min cache
_newsapi_lock = threading.Lock()
_newsapi_block_until: Optional[datetime] = None
_newsapi_last_good: dict[str, tuple[datetime, list[dict]]] = {}
_NEWSAPI_LAST_GOOD_TTL_SECONDS = 6 * 60 * 60  # 6 hours
_RSS_CONNECT_TIMEOUT_SECONDS = 4
_RSS_READ_TIMEOUT_SECONDS = 8

# RSS feeds for gold / macro news
RSS_FEEDS = {
    "google_geopolitics_gold": "https://news.google.com/rss/search?q=geopolitics+gold&hl=en-US&gl=US&ceid=US:en",
    "google_gold_price": "https://news.google.com/rss/search?q=gold+price&hl=en-US&gl=US&ceid=US:en",
    "marketwatch_topstories": "https://www.marketwatch.com/rss/topstories",
    "investing_gold": "https://www.investing.com/rss/news_14.rss",
}


class NewsDataFetcher:
    """Aggregate gold-relevant news from multiple sources."""

    _DEFAULT_KEYWORDS = [
        "gold",
        "xau",
        "bullion",
        "safe haven",
        "fed",
        "federal reserve",
        "interest rate",
        "yields",
        "treasury",
        "dollar",
        "dxy",
        "inflation",
        "cpi",
        "recession",
        "central bank",
        "opec",
        "oil",
    ]

    def _score_relevance(self, title: str, description: str, keywords: list[str]) -> int:
        blob = f"{title or ''} {description or ''}".lower()
        score = 0
        for kw in keywords:
            if kw in blob:
                score += 1
        return score

    def _rss_fallback(self, page_size: int, query: str | None = None) -> list[dict]:
        """Fallback to RSS feeds and filter to gold/macro relevance."""
        rss = self.fetch_rss()
        if not rss:
            return []

        keywords = list(self._DEFAULT_KEYWORDS)
        # If the query includes obvious keywords, include those too.
        if query:
            extra = [
                t.strip().lower()
                for t in re.split(r"\bor\b|\band\b|\(|\)|\,|\;|\|", query, flags=re.IGNORECASE)
                if t.strip()
            ]
            for t in extra:
                # Keep only short-ish terms to avoid garbage.
                if 2 <= len(t) <= 24 and any(ch.isalnum() for ch in t):
                    keywords.append(t)

        scored = []
        for a in rss:
            title = a.get("title", "")
            desc = a.get("description", "")
            scored.append((self._score_relevance(title, desc, keywords), a))

        scored.sort(key=lambda x: x[0], reverse=True)
        # Keep only articles that have *some* relevance signal; otherwise return the top items.
        filtered = [a for s, a in scored if s > 0]
        if not filtered:
            filtered = [a for _, a in scored]
        return filtered[:page_size]

    def _get_last_good(self, cache_key: str) -> list[dict]:
        """Return last known good payload for this cache key (if still fresh)."""
        with _newsapi_lock:
            hit = _newsapi_last_good.get(cache_key)
        if not hit:
            return []
        saved_at, articles = hit
        if (datetime.now() - saved_at).total_seconds() > _NEWSAPI_LAST_GOOD_TTL_SECONDS:
            return []
        return articles or []

    def _save_last_good(self, cache_key: str, articles: list[dict]) -> None:
        if not articles:
            return
        with _newsapi_lock:
            _newsapi_last_good[cache_key] = (datetime.now(), articles)

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

        cache_key = f"newsapi_{query}_{days_back}_{page_size}"
        if cache_key in _cache:
            return _cache[cache_key]

        global _newsapi_block_until
        with _newsapi_lock:
            if _newsapi_block_until and datetime.now() < _newsapi_block_until:
                # During cooldown, skip remote calls to avoid repeated 429 spam.
                last_good = self._get_last_good(cache_key)
                if last_good:
                    return last_good
                # Fall back to RSS so downstream agents still have headlines.
                return self._rss_fallback(page_size=page_size, query=query)

        from_date = (datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d")
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "from": from_date,
            "sortBy": "relevancy",
            "pageSize": page_size,
            "language": "en",
        }
        headers = {
            # Keep API key out of the query string.
            "X-Api-Key": NEWS_API_KEY,
            "User-Agent": "gold-price-predictor/1.0",
        }

        try:
            resp = requests.get(url, params=params, headers=headers, timeout=15)
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
            self._save_last_good(cache_key, cleaned)
            return cleaned
        except Exception as e:
            status_code = None
            retry_after = None
            error_detail = None
            if isinstance(e, requests.HTTPError) and getattr(e, "response", None) is not None:
                status_code = e.response.status_code
                retry_after = e.response.headers.get("Retry-After")
                try:
                    # NewsAPI returns structured errors; helpful for debugging while avoiding key leakage.
                    error_detail = e.response.json()
                except Exception:
                    error_detail = (e.response.text or "").strip()[:500]

            if status_code == 429:
                cooldown_seconds = 1800
                if retry_after and str(retry_after).isdigit():
                    cooldown_seconds = max(60, int(retry_after))
                with _newsapi_lock:
                    _newsapi_block_until = datetime.now() + timedelta(seconds=cooldown_seconds)
                logger.warning(
                    f"NewsAPI rate limit reached (429). Cooling down for {cooldown_seconds}s; using cached/RSS fallback."
                )
                last_good = self._get_last_good(cache_key)
                if last_good:
                    return last_good
                return self._rss_fallback(page_size=page_size, query=query)
            else:
                if error_detail:
                    logger.warning(
                        f"NewsAPI request failed (status={status_code}): {type(e).__name__} | detail={error_detail}"
                    )
                else:
                    logger.warning(f"NewsAPI request failed (status={status_code}): {type(e).__name__}")

            # Preserve any previously cached good data; don't overwrite with empties.
            last_good = self._get_last_good(cache_key)
            if last_good:
                return last_good
            return self._rss_fallback(page_size=page_size, query=query)

    # ------------------------------------------------------------------ #
    def fetch_rss(self) -> list[dict]:
        """Fetch articles from curated RSS feeds."""
        cache_key = "rss_all"
        if cache_key in _cache:
            return _cache[cache_key]

        articles: list[dict] = []
        for name, url in RSS_FEEDS.items():
            try:
                resp = requests.get(
                    url,
                    timeout=(_RSS_CONNECT_TIMEOUT_SECONDS, _RSS_READ_TIMEOUT_SECONDS),
                    headers={"User-Agent": "gold-price-predictor/1.0"},
                )
                resp.raise_for_status()
                feed = feedparser.parse(resp.content)
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
        # Avoid multiple remote calls per run (reduces rate-limit risk).
        combined_query = (
            "geopolitics OR war OR conflict OR sanctions OR "
            "central bank gold reserves OR federal reserve interest rate OR "
            "inflation OR recession OR BRICS OR dedollarization"
        )
        return self.fetch_newsapi(query=combined_query, days_back=days_back, page_size=25)

    # ------------------------------------------------------------------ #
    def get_all_news(self, days_back: int = 3) -> dict[str, list[dict]]:
        """Return all news grouped by category."""
        return {
            "gold_market": self.fetch_newsapi(days_back=days_back),
            "geopolitics": self.fetch_geopolitics_news(days_back),
            "rss_feeds": self.fetch_rss(),
        }
