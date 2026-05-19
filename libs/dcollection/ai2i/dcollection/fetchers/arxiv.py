"""
arXiv Atom-feed fetcher for paper-finder.

Same pattern as the PubMed + OpenAlex fetchers: synthetic corpus_id
prefix `arx:<arxiv_id>`, pre-populated fields, OriginQuery tagged
`arxiv_search`.

API docs: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import logging
from datetime import date
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

ARXIV_CORPUS_ID_PREFIX = "arx:"


def _doc_from_arxiv_entry(
    entry: dict[str, Any],
    query: str,
    rank: int,
    search_iteration: int,
) -> PaperFinderDocument | None:
    arxiv_id = entry.get("arxiv_id")
    if not arxiv_id or not isinstance(arxiv_id, str):
        return None

    corpus_id = f"{ARXIV_CORPUS_ID_PREFIX}{arxiv_id}"
    title = entry.get("title")
    abstract = entry.get("abstract")
    raw_authors = entry.get("authors") or []
    authors = [Author(name=a) for a in raw_authors if isinstance(a, str) and a.strip()]
    year = entry.get("year") if isinstance(entry.get("year"), int) else None
    doi = entry.get("doi") if isinstance(entry.get("doi"), str) else None
    primary_category = entry.get("primary_category")
    # URL preference: doi.org link if the paper has been cross-
    # referenced to a published journal article (means we link to the
    # canonical version), else arXiv's landing page from the feed,
    # else the abs/<id> URL built from the arxiv_id.
    url = (
        f"https://doi.org/{doi}"
        if doi
        else entry.get("url") or f"https://arxiv.org/abs/{arxiv_id}"
    )

    pub_date: date | None = None
    if year:
        try:
            pub_date = date(year, 1, 1)
        except ValueError:
            pub_date = None

    # arXiv is preprint-by-default. We add "preprint" as a publication
    # type so the downstream review-filter doesn't accidentally tag
    # these as reviews (only happens via title heuristic now).
    publication_types = entry.get("publication_types") or ["preprint"]

    origin = OriginQuery(
        query_type="arxiv_search",
        provider="arxiv",
        dataset="arxiv",
        variant=primary_category if isinstance(primary_category, str) else None,
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
        venue="arXiv",
        publication_types=publication_types,
        publication_date=pub_date,
        snippets=[],
        citation_contexts=[],
        origins=[origin],
    )


async def fetch_from_arxiv_search(
    queries: list[str],
    search_iteration: int,
    top_k: int,
    context: DocumentCollectionContext,
    time_range: ExtractedYearlyTimeRange | None = None,
) -> list[Document]:
    """Run arXiv search for each query and return merged Documents."""
    client = context.arxiv_client
    if client is None:
        logger.info("[arxiv_fetcher] No arXiv client on context; skipping arm.")
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
            logger.error(f"[arxiv_fetcher] Query failed (continuing): {query[:80]!r} → {e}")
            continue

        for rank, entry in enumerate(raw, start=1):
            doc = _doc_from_arxiv_entry(
                entry=entry,
                query=query,
                rank=rank,
                search_iteration=search_iteration,
            )
            if doc is not None:
                out.append(doc)

    logger.info(
        f"[arxiv_fetcher] Returned {len(out)} doc(s) across {len(queries)} query/queries "
        f"(top_k={top_k}, iteration={search_iteration})"
    )
    return out
