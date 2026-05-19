"""
Google Scholar fetcher (via SerpAPI) for paper-finder.

Synthetic corpus_id prefix `gs:<query-hash>-<rank>`. Scholar doesn't
expose a stable per-paper ID, so we synthesize one from the search
query + rank position to avoid cross-query collisions.
"""

from __future__ import annotations

import hashlib
import logging
from typing import Any

from ai2i.dcollection import PaperFinderDocument
from ai2i.dcollection.data_access_context import DocumentCollectionContext
from ai2i.dcollection.interface.collection import Document
from ai2i.dcollection.interface.document import (
    Author,
    ExtractedYearlyTimeRange,
    OriginQuery,
)

logger = logging.getLogger(__name__)

SCHOLAR_CORPUS_ID_PREFIX = "gs:"


def _query_fingerprint(query: str) -> str:
    """Short stable hash of the query so each Scholar result gets a
    unique corpus_id without colliding across different searches."""
    return hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]


def _doc_from_scholar_result(
    result: dict[str, Any],
    query: str,
    rank: int,
    search_iteration: int,
) -> PaperFinderDocument | None:
    title = result.get("title")
    if not title or not isinstance(title, str):
        return None

    # No stable per-paper ID from Scholar. Use a (query-hash, rank) pair
    # so reruns of the same query stay stable but different queries
    # produce different IDs.
    corpus_id = f"{SCHOLAR_CORPUS_ID_PREFIX}{_query_fingerprint(query)}-{rank}"

    abstract = result.get("abstract")  # SerpAPI's snippet, not a full abstract
    raw_authors = result.get("authors") or []
    authors = [Author(name=a) for a in raw_authors if isinstance(a, str) and a.strip()]
    year = result.get("year") if isinstance(result.get("year"), int) else None
    doi = result.get("doi") if isinstance(result.get("doi"), str) else None
    raw_url = result.get("url") if isinstance(result.get("url"), str) else None

    # URL preference: an authoritative doi.org link beats the raw
    # publisher link Scholar surfaced (publisher links sometimes
    # redirect through paywalled aggregators). Fall back to the raw
    # URL and finally to a Scholar search if neither is available.
    if doi:
        url = f"https://doi.org/{doi}"
    elif raw_url:
        url = raw_url
    else:
        from urllib.parse import quote_plus

        url = f"https://scholar.google.com/scholar?q={quote_plus(title)}"

    publication_types = result.get("publication_types") or []
    if not isinstance(publication_types, list):
        publication_types = []

    origin = OriginQuery(
        query_type="scholar_search",
        provider="scholar",
        dataset="serpapi",
        variant=None,
        query=query,
        iteration=search_iteration,
        ranks=[rank],
    )

    return PaperFinderDocument(
        corpus_id=corpus_id,
        url=url,
        title=title,
        year=year,
        authors=authors or None,
        abstract=abstract,
        publication_types=publication_types or None,
        snippets=[],
        citation_contexts=[],
        origins=[origin],
    )


async def fetch_from_scholar_search(
    queries: list[str],
    search_iteration: int,
    top_k: int,
    context: DocumentCollectionContext,
    time_range: ExtractedYearlyTimeRange | None = None,
) -> list[Document]:
    """Search Google Scholar (via SerpAPI) for each query."""
    client = context.scholar_client
    if client is None or not client.is_available():
        # Disabled or no API key — silently skip. The aggregator handles
        # zero-result returns gracefully.
        return []

    out: list[Document] = []
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            continue
        try:
            raw = await client.search(
                query=query,
                max_results=top_k,
                time_range_start=time_range.start if time_range and time_range.start else None,
                time_range_end=time_range.end if time_range and time_range.end else None,
            )
        except Exception as e:
            logger.error(f"[scholar_fetcher] Query failed (continuing): {query[:80]!r} → {e}")
            continue

        for rank, result in enumerate(raw, start=1):
            doc = _doc_from_scholar_result(
                result=result,
                query=query,
                rank=rank,
                search_iteration=search_iteration,
            )
            if doc is not None:
                out.append(doc)

    logger.info(
        f"[scholar_fetcher] Returned {len(out)} doc(s) across {len(queries)} query/queries "
        f"(top_k={top_k}, iteration={search_iteration})"
    )
    return out
