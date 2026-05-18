"""
OpenAlex search fetcher.

Mirrors the shape of `fetch_from_vespa_dense_retrieval` and
`s2_paper_search` so the calling agent code can treat OpenAlex as just
another retrieval arm in `_run_initial_retrieval`'s parallel
fan-out.

Notes on integration with the S2-centric document model:

- `PaperFinderDocument.corpus_id` is required (`Field()`) AND most
  metadata fields are lazy-loaded via the `from_s2` REST loader,
  keyed on `int(corpus_id)`. OpenAlex works carry no native S2
  corpus_id, so we tag ours with a synthetic prefix `oa:<work-id>`
  (e.g. `oa:W2741809807`) so they never collide with numeric S2
  IDs.

- To keep these docs usable without triggering S2 enrichment (which
  would 4xx on the synthetic ID), we pre-populate every standard
  field from the OpenAlex response. The PaperFinderDocument
  `__init__` adds every kwarg to `_loaded_fields`, which makes the
  dynamic-field loaders skip those fields. The `from_s2` loader
  also got a defensive guard for non-numeric corpus_ids so any code
  path that does try to enrich gracefully no-ops.

- OpenAlex docs don't carry snippets. They flow through Cohere
  rerank (abstract-only) and relevance judgement, but not through
  snippet snowball expansion.

API docs: https://docs.openalex.org/api-entities/works
"""

from __future__ import annotations

import logging
from datetime import date, datetime
from typing import Any

from ai2i.dcollection import PaperFinderDocument
from ai2i.dcollection.data_access_context import DocumentCollectionContext
from ai2i.dcollection.external_api.openalex import (
    reconstruct_abstract_from_inverted_index,
)
from ai2i.dcollection.interface.collection import Document
from ai2i.dcollection.interface.document import (
    Author,
    ExtractedYearlyTimeRange,
    Journal,
    OriginQuery,
)

logger = logging.getLogger(__name__)


# Prefix that marks a corpus_id as OpenAlex-sourced (not a real S2 ID).
# Anything starting with this MUST be skipped by S2-keyed loaders.
OPENALEX_CORPUS_ID_PREFIX = "oa:"


def _is_openalex_corpus_id(corpus_id: str | None) -> bool:
    return isinstance(corpus_id, str) and corpus_id.startswith(OPENALEX_CORPUS_ID_PREFIX)


def _extract_openalex_work_id(openalex_id_url: str | None) -> str | None:
    """`https://openalex.org/W2741809807` → `W2741809807`."""
    if not isinstance(openalex_id_url, str) or not openalex_id_url:
        return None
    return openalex_id_url.rsplit("/", 1)[-1] or None


def _strip_doi_prefix(doi_url: str | None) -> str | None:
    """`https://doi.org/10.x/y` → `10.x/y` so dedupe-by-DOI matches S2 docs."""
    if not isinstance(doi_url, str) or not doi_url:
        return None
    lower = doi_url.lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "http://dx.doi.org/"):
        if lower.startswith(prefix):
            return doi_url[len(prefix):]
    return doi_url


def _parse_authors(authorships: Any) -> list[Author]:
    if not isinstance(authorships, list):
        return []
    out: list[Author] = []
    for a in authorships:
        if not isinstance(a, dict):
            continue
        author = a.get("author") or {}
        name = author.get("display_name")
        if isinstance(name, str) and name:
            out.append(Author(name=name))
    return out


def _parse_publication_date(value: Any) -> date | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError:
        # Some OpenAlex records only have year-resolution dates.
        try:
            return datetime.strptime(value, "%Y").date()
        except ValueError:
            return None


def _resolve_url(work: dict[str, Any], doi: str | None, openalex_id_url: str | None) -> str:
    """Best landing page → DOI → OpenAlex work page (last resort)."""
    primary = work.get("primary_location") or {}
    landing = primary.get("landing_page_url")
    if isinstance(landing, str) and landing:
        return landing
    if doi:
        return f"https://doi.org/{doi}"
    if isinstance(openalex_id_url, str) and openalex_id_url:
        return openalex_id_url
    return ""


def _doc_from_openalex_work(
    work: dict[str, Any],
    query: str,
    rank: int,
    search_iteration: int,
) -> PaperFinderDocument | None:
    """Map a single OpenAlex result into a PaperFinderDocument.

    Returns None if the work has no resolvable identifier so we never
    emit zombie docs with an empty `corpus_id`.
    """
    openalex_id_url = work.get("id")
    work_id = _extract_openalex_work_id(openalex_id_url if isinstance(openalex_id_url, str) else None)
    if not work_id:
        logger.warning(f"[openalex_fetcher] Skipping work with no resolvable id: {work}")
        return None

    corpus_id = f"{OPENALEX_CORPUS_ID_PREFIX}{work_id}"
    doi = _strip_doi_prefix(work.get("doi") if isinstance(work.get("doi"), str) else None)

    title = work.get("title") if isinstance(work.get("title"), str) else None
    publication_year = work.get("publication_year") if isinstance(work.get("publication_year"), int) else None
    publication_date = _parse_publication_date(work.get("publication_date"))
    authors = _parse_authors(work.get("authorships"))
    abstract = reconstruct_abstract_from_inverted_index(work.get("abstract_inverted_index"))

    cited_by_count = work.get("cited_by_count") if isinstance(work.get("cited_by_count"), int) else None

    # OpenAlex 'type' maps to S2-style publication_types. We retain the
    # raw type as a single-element list since downstream filters check
    # for 'review' substring; this preserves that semantics.
    raw_type = work.get("type") if isinstance(work.get("type"), str) else None
    publication_types = [raw_type] if raw_type else None

    primary_location = work.get("primary_location") if isinstance(work.get("primary_location"), dict) else {}
    source_meta = primary_location.get("source") if isinstance(primary_location.get("source"), dict) else {}
    journal_name = source_meta.get("display_name") if isinstance(source_meta.get("display_name"), str) else None
    journal = Journal(name=journal_name) if journal_name else None
    venue = journal_name  # mirror s2_paper_to_document's pattern

    fields_of_study = _parse_fields_of_study(work.get("concepts"))

    url = _resolve_url(work, doi, openalex_id_url if isinstance(openalex_id_url, str) else None)

    origin = OriginQuery(
        query_type="openalex_search",
        provider="openalex",
        dataset="works",
        variant=None,
        query=query,
        iteration=search_iteration,
        ranks=[rank],
    )

    # Pre-populate every field the downstream pipeline needs. Keys
    # passed to __init__ are auto-marked as loaded, so the from_s2
    # dynamic-field loaders won't fire on this doc (they would 4xx
    # anyway given the synthetic corpus_id).
    return PaperFinderDocument(
        corpus_id=corpus_id,
        url=url,
        title=title,
        year=publication_year,
        authors=authors or None,
        abstract=abstract or None,
        venue=venue,
        publication_types=publication_types,
        fields_of_study=fields_of_study,
        citation_count=cited_by_count,
        journal=journal,
        publication_date=publication_date,
        snippets=None,
        origins=[origin],
    )


def _parse_fields_of_study(concepts: Any) -> list[str] | None:
    """OpenAlex `concepts: [{display_name, level, score, ...}]` → top-N names."""
    if not isinstance(concepts, list):
        return None
    names: list[str] = []
    for c in concepts:
        if not isinstance(c, dict):
            continue
        name = c.get("display_name")
        if isinstance(name, str) and name:
            names.append(name)
        if len(names) >= 5:
            break
    return names or None


async def fetch_from_openalex_search(
    queries: list[str],
    search_iteration: int,
    top_k: int,
    context: DocumentCollectionContext,
    time_range: ExtractedYearlyTimeRange | None = None,
    fields_of_study: list[str] | None = None,
) -> list[Document]:
    """Run an OpenAlex `/works` search for each query and merge results.

    Mirrors the contract of `fetch_from_vespa_dense_retrieval`: takes a
    list of queries (typically just one for the broad search arm,
    multiple for multi-query strategies), returns a flat list of
    Documents with `origins` set so downstream rank fusion can attribute
    each doc back to its source query + rank.
    """
    openalex_client = context.openalex_client
    if openalex_client is None:
        logger.info("[openalex_fetcher] No OpenAlex client on context; skipping arm.")
        return []

    out: list[Document] = []
    for query in queries:
        if not isinstance(query, str) or not query.strip():
            continue
        try:
            raw_results = await openalex_client.search_works(
                query=query,
                per_page=top_k,
                time_range_start=time_range.start if time_range and time_range.start else None,
                time_range_end=time_range.end if time_range and time_range.end else None,
                fields_of_study=fields_of_study,
            )
        except Exception as e:
            # Don't sink the whole arm if one query throws; log + continue
            # so a single bad query doesn't zero out the OpenAlex
            # contribution.
            logger.error(f"[openalex_fetcher] Query failed (continuing): {query[:80]!r} → {e}")
            continue

        for rank, work in enumerate(raw_results, start=1):
            if not isinstance(work, dict):
                continue
            doc = _doc_from_openalex_work(
                work=work,
                query=query,
                rank=rank,
                search_iteration=search_iteration,
            )
            if doc is not None:
                out.append(doc)

    logger.info(
        f"[openalex_fetcher] Returned {len(out)} doc(s) across {len(queries)} query/queries "
        f"(top_k={top_k}, iteration={search_iteration})"
    )
    return out
