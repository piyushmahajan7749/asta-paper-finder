"""
Async Google Scholar client (via SerpAPI).

Google Scholar doesn't have an official API. SerpAPI scrapes the public
Scholar pages and exposes them as a structured JSON endpoint. Paid
service; needs `SERPAPI_API_KEY`. We register this arm as
`enabled: false` by default so deployments without the key don't fail.

SerpAPI docs: https://serpapi.com/google-scholar-api

Returned metadata is less rich than PubMed/arXiv:
  - title: always present
  - snippet: short description (not a full abstract)
  - authors: only when SerpAPI extracted them from the byline
  - year: parsed from publication_info.summary best-effort
  - doi: only when the paper's landing-page URL is a DOI link
"""

from __future__ import annotations

import logging
import re
from typing import Any

import httpx
from httpx import HTTPStatusError, NetworkError, TimeoutException

logger = logging.getLogger(__name__)


SERPAPI_URL = "https://serpapi.com/search.json"


class AsyncScholarClient:
    def __init__(self, api_key: str | None = None, timeout: int = 30) -> None:
        self.api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.timeout = timeout
        if self.api_key:
            logger.info("[scholar] Client constructed with SERPAPI_API_KEY")
        else:
            logger.warning(
                "[scholar] No SERPAPI_API_KEY set. Scholar arm will return [] on every call. "
                "Set SERPAPI_API_KEY in Azure App Service config to enable."
            )

    def is_available(self) -> bool:
        return self.api_key is not None

    async def search(
        self,
        query: str,
        max_results: int = 20,
        time_range_start: int | None = None,
        time_range_end: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search Google Scholar via SerpAPI."""
        if not self.api_key:
            return []
        if not query or not query.strip():
            return []

        max_results = max(1, min(max_results, 20))  # SerpAPI caps at 20

        params: dict[str, Any] = {
            "engine": "google_scholar",
            "q": query.strip(),
            "api_key": self.api_key,
            "num": max_results,
        }
        if time_range_start:
            params["as_ylo"] = time_range_start
        if time_range_end:
            params["as_yhi"] = time_range_end

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(SERPAPI_URL, params=params)
                if r.status_code == 429:
                    logger.warning(
                        "[scholar] SerpAPI rate-limited (429); returning empty for this call."
                    )
                    return []
                r.raise_for_status()
                payload = r.json()
        except HTTPStatusError as e:
            logger.error(
                f"[scholar] SerpAPI HTTP {e.response.status_code}: {e.response.text[:200]}"
            )
            return []
        except (NetworkError, TimeoutException) as e:
            logger.error(f"[scholar] SerpAPI network/timeout: {e}")
            return []

        organic = payload.get("organic_results") or []
        if not isinstance(organic, list):
            return []

        return [_parse_organic_result(r, idx) for idx, r in enumerate(organic) if isinstance(r, dict)]


# "Author1, Author2 - Journal Name, 2024 - publisher.com"
# We split on `-` and pick the year from the middle segment.
_YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")


def _parse_organic_result(r: dict[str, Any], idx: int) -> dict[str, Any]:
    title = r.get("title") if isinstance(r.get("title"), str) else None
    snippet = r.get("snippet") if isinstance(r.get("snippet"), str) else None
    link = r.get("link") if isinstance(r.get("link"), str) else None

    pub_info = r.get("publication_info") or {}
    pub_summary = pub_info.get("summary") if isinstance(pub_info, dict) else None

    authors_list: list[str] = []
    if isinstance(pub_info, dict):
        raw_authors = pub_info.get("authors")
        if isinstance(raw_authors, list):
            for a in raw_authors:
                if isinstance(a, dict) and isinstance(a.get("name"), str):
                    authors_list.append(a["name"])
        elif isinstance(pub_summary, str):
            # Fallback: parse the comma-separated authors from the
            # summary's first segment ("J Smith, A Doe - Nature, 2024").
            head = pub_summary.split(" - ", 1)[0]
            authors_list = [s.strip() for s in head.split(",") if s.strip()]

    year: int | None = None
    if isinstance(pub_summary, str):
        m = _YEAR_RE.search(pub_summary)
        if m:
            try:
                year = int(m.group(0))
            except ValueError:
                year = None

    # DOI: Scholar doesn't surface it as a field, but some result links
    # ARE doi.org URLs. Pull from there when possible.
    doi: str | None = None
    if isinstance(link, str) and "doi.org/" in link.lower():
        try:
            doi = link.split("doi.org/", 1)[1].strip().lower() or None
        except Exception:
            doi = None

    return {
        "scholar_idx": idx,  # stable per-query identifier
        "title": title,
        "abstract": snippet,  # SerpAPI gives a snippet, not full abstract
        "authors": authors_list,
        "year": year,
        "url": link,
        "doi": doi,
        # Scholar doesn't tag review articles. Downstream
        # detect_review_article will still catch them by title
        # heuristic + abstract opener.
        "publication_types": [],
    }
