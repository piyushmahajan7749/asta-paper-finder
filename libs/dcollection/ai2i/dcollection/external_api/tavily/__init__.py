"""
Async Tavily web search client.

Tavily is a general web search API with an "advanced" mode tuned for
research workflows. Not strictly academic — catches blog posts,
preprints, whitepapers, conference abstracts, and news. We include it
in the lit-scout pool as a recency net: very recent work that hasn't
hit PubMed / OpenAlex yet often shows up on a personal/lab website
that Tavily indexes.

Paid; needs `TAVILY_API_KEY`. Disabled by default; deployments can
opt in via config.

Tavily docs: https://docs.tavily.com/
"""

from __future__ import annotations

import logging
from typing import Any

import httpx
from httpx import HTTPStatusError, NetworkError, TimeoutException

logger = logging.getLogger(__name__)


TAVILY_SEARCH_URL = "https://api.tavily.com/search"


class AsyncTavilyClient:
    def __init__(self, api_key: str | None = None, timeout: int = 20) -> None:
        self.api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.timeout = timeout
        if self.api_key:
            logger.info("[tavily] Client constructed with TAVILY_API_KEY")
        else:
            logger.warning(
                "[tavily] No TAVILY_API_KEY set. Tavily arm will return [] on every call. "
                "Set TAVILY_API_KEY in Azure App Service config to enable."
            )

    def is_available(self) -> bool:
        return self.api_key is not None

    async def search(
        self,
        query: str,
        max_results: int = 15,
        include_domains: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Run a Tavily search and return result dicts."""
        if not self.api_key:
            return []
        if not query or not query.strip():
            return []

        max_results = max(1, min(max_results, 20))

        body: dict[str, Any] = {
            "api_key": self.api_key,
            "query": query.strip(),
            "max_results": max_results,
            # "advanced" is Tavily's research-tuned mode (more
            # comprehensive crawl + better content extraction). Worth
            # the extra latency for a lit-scout flow.
            "search_depth": "advanced",
            # Don't include a generated answer; we want raw results
            # only.
            "include_answer": False,
            "include_raw_content": False,
        }
        if include_domains:
            body["include_domains"] = include_domains

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.post(TAVILY_SEARCH_URL, json=body)
                if r.status_code == 429:
                    logger.warning("[tavily] rate-limited (429); returning empty.")
                    return []
                r.raise_for_status()
                payload = r.json()
        except HTTPStatusError as e:
            logger.error(f"[tavily] HTTP {e.response.status_code}: {e.response.text[:200]}")
            return []
        except (NetworkError, TimeoutException) as e:
            logger.error(f"[tavily] network/timeout: {e}")
            return []

        results = payload.get("results") or []
        if not isinstance(results, list):
            return []

        return [
            _parse_result(r, idx) for idx, r in enumerate(results) if isinstance(r, dict)
        ]


def _parse_result(r: dict[str, Any], idx: int) -> dict[str, Any]:
    title = r.get("title") if isinstance(r.get("title"), str) else None
    content = r.get("content") if isinstance(r.get("content"), str) else None
    url = r.get("url") if isinstance(r.get("url"), str) else None

    # Tavily sometimes returns `published_date` as YYYY-MM-DD; year-only
    # papers from a personal/lab page might not have it at all.
    year: int | None = None
    pubdate = r.get("published_date")
    if isinstance(pubdate, str) and len(pubdate) >= 4 and pubdate[:4].isdigit():
        try:
            year = int(pubdate[:4])
        except ValueError:
            year = None

    # No author parsing - Tavily returns free-text content blocks, not
    # structured byline. Empty list signals "unknown" downstream.
    return {
        "tavily_idx": idx,
        "title": title,
        "abstract": content,  # Tavily's `content` IS the extracted summary
        "authors": [],
        "year": year,
        "url": url,
        # No DOI extraction; would require LLM parsing of content body.
        "doi": None,
        "publication_types": [],
    }
