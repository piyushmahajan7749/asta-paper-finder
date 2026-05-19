"""
Tavily web-search fetcher for paper-finder.

Synthetic corpus_id prefix `tav:<query-hash>-<rank>`. Tavily indexes
arbitrary web content (blogs, lab pages, preprints, news) so the
returned items are not always papers in the traditional sense — but
they're invaluable for catching very recent work that hasn't hit the
academic indexes yet.
"""

from __future__ import annotations

import hashlib
import logging
from datetime import date
from typing import Any

from ai2i.dcollection import PaperFinderDocument
from ai2i.dcollection.data_access_context import DocumentCollectionContext
from ai2i.dcollection.interface.collection import Document
from ai2i.dcollection.interface.document import (
    ExtractedYearlyTimeRange,
    OriginQuery,
)

logger = logging.getLogger(__name__)

TAVILY_CORPUS_ID_PREFIX = "tav:"


def _query_fingerprint(query: str) -> str:
    return hashlib.sha1(query.encode("utf-8")).hexdigest()[:10]


def _doc_from_tavily_result(
    result: dict[str, Any],
    query: str,
    rank: int,
    search_iteration: int,
) -> PaperFinderDocument | None:
    title = result.get("title")
    if not title or not isinstance(title, str):
        return None
    url = result.get("url") if isinstance(result.get("url"), str) else None
    if not url:
        return None  # Web result without a URL is useless; drop it.

    corpus_id = f"{TAVILY_CORPUS_ID_PREFIX}{_query_fingerprint(query)}-{rank}"
    abstract = result.get("abstract")  # Tavily's `content` field
    year = result.get("year") if isinstance(result.get("year"), int) else None

    pub_date: date | None = None
    if year:
        try:
            pub_date = date(year, 1, 1)
        except ValueError:
            pub_date = None

    origin = OriginQuery(
        query_type="tavily_search",
        provider="tavily",
        dataset="web",
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
        authors=None,  # Tavily doesn't return structured bylines
        abstract=abstract,
        # No journal/venue from Tavily results - the "publication" is
        # just a web page.
        publication_date=pub_date,
        # No publication_types from web search. Title heuristic in the
        # downstream review-filter still catches "review of …" style
        # pages.
        snippets=[],
        citation_contexts=[],
        origins=[origin],
    )


async def fetch_from_tavily_search(
    queries: list[str],
    search_iteration: int,
    top_k: int,
    context: DocumentCollectionContext,
    time_range: ExtractedYearlyTimeRange | None = None,
) -> list[Document]:
    """Search Tavily for each query and return merged Documents."""
    # time_range unused for Tavily today - the API doesn't expose a
    # date-range filter we can plumb through cleanly. Accept the
    # parameter for shape parity with the other fetchers.
    _ = time_range

    client = context.tavily_client
    if client is None or not client.is_available():
        return []

    out: list[Document] = []
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            continue
        try:
            raw = await client.search(query=query, max_results=top_k)
        except Exception as e:
            logger.error(f"[tavily_fetcher] Query failed (continuing): {query[:80]!r} → {e}")
            continue

        for rank, result in enumerate(raw, start=1):
            doc = _doc_from_tavily_result(
                result=result,
                query=query,
                rank=rank,
                search_iteration=search_iteration,
            )
            if doc is not None:
                out.append(doc)

    logger.info(
        f"[tavily_fetcher] Returned {len(out)} doc(s) across {len(queries)} query/queries "
        f"(top_k={top_k}, iteration={search_iteration})"
    )
    return out
