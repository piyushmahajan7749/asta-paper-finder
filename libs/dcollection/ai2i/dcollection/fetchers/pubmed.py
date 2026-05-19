"""
PubMed (NCBI eutils) fetcher for paper-finder.

Follows the OpenAlex pattern: hit the client, map results to
PaperFinderDocument with synthetic corpus_id (`pm:<pmid>`), pre-
populate every standard field so the dynamic-field loaders don't
fire on these docs (their corpus_id isn't a numeric S2 id, so a
from_s2 lookup would 4xx anyway).

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
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
    Journal,
    OriginQuery,
)

logger = logging.getLogger(__name__)

PUBMED_CORPUS_ID_PREFIX = "pm:"


def _doc_from_pubmed_article(
    article: dict[str, Any],
    query: str,
    rank: int,
    search_iteration: int,
) -> PaperFinderDocument | None:
    pmid = article.get("pmid")
    if not pmid or not isinstance(pmid, str):
        return None

    corpus_id = f"{PUBMED_CORPUS_ID_PREFIX}{pmid}"
    title = article.get("title")
    abstract = article.get("abstract")
    raw_authors = article.get("authors") or []
    authors = [Author(name=a) for a in raw_authors if isinstance(a, str) and a.strip()]
    year = article.get("year") if isinstance(article.get("year"), int) else None
    doi = article.get("doi") if isinstance(article.get("doi"), str) else None
    journal_name = article.get("journal") if isinstance(article.get("journal"), str) else None
    publication_types = article.get("publication_types") or []
    if not isinstance(publication_types, list):
        publication_types = []

    # Prefer the DOI URL when we have it (publisher's authoritative
    # landing page). Fall back to the PubMed abstract page.
    url = (
        f"https://doi.org/{doi}"
        if doi
        else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
    )

    pub_date: date | None = None
    if year:
        try:
            pub_date = date(year, 1, 1)
        except ValueError:
            pub_date = None

    origin = OriginQuery(
        query_type="pubmed_search",
        provider="pubmed",
        dataset="pubmed",
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
        venue=journal_name,
        publication_types=publication_types or None,
        journal=Journal(name=journal_name) if journal_name else None,
        publication_date=pub_date,
        snippets=[],
        citation_contexts=[],
        origins=[origin],
    )


async def fetch_from_pubmed_search(
    queries: list[str],
    search_iteration: int,
    top_k: int,
    context: DocumentCollectionContext,
    time_range: ExtractedYearlyTimeRange | None = None,
) -> list[Document]:
    """Fan out PubMed searches for each query, return merged Documents.

    Matches the shape of `fetch_from_openalex_search` so the calling
    agent treats every arm uniformly. PubMed lacks a 1:1 concept-filter
    analog to S2's `fields_of_study`, so we drop that filter at the
    boundary.
    """
    client = context.pubmed_client
    if client is None:
        logger.info("[pubmed_fetcher] No PubMed client on context; skipping arm.")
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
            logger.error(f"[pubmed_fetcher] Query failed (continuing): {query[:80]!r} → {e}")
            continue

        for rank, article in enumerate(raw, start=1):
            doc = _doc_from_pubmed_article(
                article=article,
                query=query,
                rank=rank,
                search_iteration=search_iteration,
            )
            if doc is not None:
                out.append(doc)

    logger.info(
        f"[pubmed_fetcher] Returned {len(out)} doc(s) across {len(queries)} query/queries "
        f"(top_k={top_k}, iteration={search_iteration})"
    )
    return out
