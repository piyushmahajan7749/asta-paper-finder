from __future__ import annotations

import logging
from collections import defaultdict
from typing import Any, Iterable, Sequence

from semanticscholar import AsyncSemanticScholar

from ai2i.dcollection import PaperFinderDocument
from ai2i.dcollection.caching.cache import SubsetCache
from ai2i.dcollection.collection import PaperFinderDocumentCollection
from ai2i.dcollection.data_access_context import DocumentCollectionContext, SubsetCacheInterface
from ai2i.dcollection.external_api.dense.vespa import VespaRetriever
from ai2i.dcollection.external_api.openalex import AsyncOpenAlexClient
from ai2i.dcollection.fetchers.dense import DenseDataset, fetch_from_vespa_dense_retrieval
from ai2i.dcollection.fetchers.openalex import fetch_from_openalex_search
from ai2i.dcollection.fetchers.s2 import (
    s2_by_author,
    s2_fetch_citing_papers,
    s2_paper_search,
    s2_papers_by_title,
)
from ai2i.dcollection.interface.collection import (
    BASIC_FIELDS,
    S2_FIELDS,
    BaseDocumentCollectionFactory,
    Document,
    DocumentCollection,
)
from ai2i.dcollection.interface.document import (
    CorpusId,
    DocumentFieldName,
    ExtractedYearlyTimeRange,
)

logger = logging.getLogger(__name__)


class DocumentCollectionFactory(BaseDocumentCollectionFactory):
    def __init__(
        self,
        /,
        s2_api_key: str | None = None,
        s2_api_timeout: int = 60,
        s2_max_concurrency: int = 10,
        vespa_max_concurrency: int = 10,
        cache_ttl: int = 600,
        cache_is_enabled: bool = True,
        force_deterministic: bool = False,
        openalex_mailto: str | None = None,
        openalex_timeout: int = 15,
    ):
        super().__init__()
        # Fail loud when the S2 API key is missing or blank. The
        # downstream `snippet/search` endpoint (called by VespaRetriever)
        # returns 403 Forbidden without a valid key, which used to
        # bubble up to the agent layer as a generic "BroadSearchAgent
        # failed to respond" message and was easy to misdiagnose as a
        # Vespa/AWS issue. Logging here is the only signal an operator
        # gets at boot time; the client is still built (no key) so
        # local dev with a free tier keeps working, but every dense
        # retrieval will throw a clearer error from _validate_response.
        normalized_s2_key = s2_api_key.strip() if isinstance(s2_api_key, str) else None
        if not normalized_s2_key:
            logger.error(
                "S2_API_KEY is missing or empty. PaperFinder will fall back to "
                "the Semantic Scholar free tier, which does NOT include the "
                "snippet-search endpoint - expect 403 Forbidden on every dense "
                "retrieval. Set the S2_API_KEY application setting (Azure App "
                "Service → Configuration → Application settings) or .env.secret "
                "to a valid key from "
                "https://www.semanticscholar.org/product/api#api-key and "
                "restart the service."
            )
        s2_client = (
            AsyncSemanticScholar(timeout=s2_api_timeout, api_key=normalized_s2_key)
            if normalized_s2_key
            else AsyncSemanticScholar(timeout=s2_api_timeout)
        )
        vespa_client = VespaRetriever(
            s2_client=s2_client,
            timeout=s2_api_timeout,
        )
        # OpenAlex client is always constructed - if `openalex_mailto`
        # is missing the client falls back to the common pool (with a
        # boot-time warning). That keeps the OpenAlex retrieval arm
        # functional out of the box; an operator can opt into the
        # polite pool later just by setting the env var, no code
        # change needed.
        openalex_client = AsyncOpenAlexClient(
            mailto=openalex_mailto,
            timeout=openalex_timeout,
        )
        cache = SubsetCache(
            ttl=cache_ttl,
            is_enabled=cache_is_enabled,
            force_deterministic=force_deterministic,
        )
        self._context = DocumentCollectionContext(
            s2_client=s2_client,
            s2_max_concurrency=s2_max_concurrency,
            vespa_client=vespa_client,
            vespa_max_concurrency=vespa_max_concurrency,
            openalex_client=openalex_client,
            cache=cache,
            force_deterministic=force_deterministic,
        )

    def s2_client(self) -> AsyncSemanticScholar:
        return self._context.s2_client

    def cache(self) -> SubsetCacheInterface:
        return self._context.cache

    def context(self) -> DocumentCollectionContext:
        return self._context

    def from_ids(self, corpus_ids: list[CorpusId]) -> DocumentCollection:
        """Create a document collection from a list of corpus IDs."""
        return self.from_docs([PaperFinderDocument(corpus_id=corpus_id) for corpus_id in corpus_ids])

    def from_docs(
        self,
        documents: Sequence[Document],
        computed_fields: dict[DocumentFieldName, Any] | None = None,
    ) -> DocumentCollection:
        """Create a document collection from a list of documents."""
        docs_for_fuse: dict[CorpusId, list[Document]] = defaultdict(list)
        for doc in documents:
            docs_for_fuse[doc.corpus_id].append(doc)
        fused_docs = []
        for doc_group in docs_for_fuse.values():
            fused_docs.append(doc_group[0].fuse(*doc_group[1:]))
        return (
            PaperFinderDocumentCollection(documents=list(fused_docs), factory=self)
            if not computed_fields
            else PaperFinderDocumentCollection(
                documents=list(fused_docs),
                computed_fields=computed_fields or {},
                factory=self,
            )
        )

    def empty(self) -> DocumentCollection:
        """Create an empty document collection."""
        return self.from_docs([])

    def merge(self, collections: Iterable[DocumentCollection]) -> DocumentCollection:
        return PaperFinderDocumentCollection(factory=self).merged(*collections)

    def from_dict(self, params: dict[str, Any]) -> PaperFinderDocumentCollection:
        rest = {k: v for k, v in params.items() if k != "documents"}
        if "documents" in params:
            documents = [PaperFinderDocument.from_dict(d) for d in params["documents"]]
        else:
            documents = []
        return PaperFinderDocumentCollection(documents=documents, factory=self, **rest)

    async def from_s2_by_author(
        self, authors_profiles: list[list[Any]], limit: int, inserted_before: str | None
    ) -> DocumentCollection:
        """Create a document collection from S2 by author."""
        context = self._context
        if len(authors_profiles) == 1:
            documents = await s2_by_author(authors_profiles[0], context, inserted_before)
        else:
            raise NotImplementedError

        collection = self.from_docs(documents=documents)
        return await collection.take(limit).with_fields(BASIC_FIELDS)

    async def from_s2_by_title(
        self,
        query: str,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        inserted_before: str | None = None,
    ) -> DocumentCollection:
        """Create a document collection from S2 by title."""
        documents = await s2_papers_by_title(
            query,
            time_range=time_range,
            venues=venues,
            context=self._context,
            inserted_before=inserted_before,
        )
        collection = self.from_docs(documents=documents)
        return await collection.with_fields(BASIC_FIELDS)

    async def from_s2_search(
        self,
        query: str,
        limit: int,
        search_iteration: int = 1,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        fields_of_study: list[str] | None = None,
        min_citations: int | None = None,
        fields: list[DocumentFieldName] | None = None,
        inserted_before: str | None = None,
    ) -> DocumentCollection:
        """Create a document collection from S2 search."""
        documents = await s2_paper_search(
            query,
            search_iteration=search_iteration,
            time_range=time_range,
            venues=venues,
            fields_of_study=fields_of_study,
            min_citations=min_citations,
            total_limit=limit,
            context=self._context,
            inserted_before=inserted_before,
        )
        collection = self.from_docs(documents=documents)
        return await collection.with_fields(fields or BASIC_FIELDS)

    async def from_s2_citing_papers(
        self,
        corpus_id: CorpusId,
        search_iteration: int = 1,
        total_limit: int = 1000,
        inserted_before: str | None = None,
    ) -> DocumentCollection:
        """Create a document collection from S2 citing papers."""
        documents = await s2_fetch_citing_papers(
            corpus_id,
            search_iteration=search_iteration,
            context=self._context,
            total_limit=total_limit,
            inserted_before=inserted_before,
        )
        collection = self.from_docs(documents=documents)
        return await collection.with_fields(BASIC_FIELDS)

    async def from_dense_retrieval(
        self,
        queries: list[str],
        search_iteration: int,
        dataset: DenseDataset,
        top_k: int,
        time_range: ExtractedYearlyTimeRange | None = None,
        venues: list[str] | None = None,
        authors: list[str] | None = None,
        corpus_ids: list[CorpusId] | None = None,
        fields_of_study: list[str] | None = None,
        inserted_before: str | None = None,
    ) -> DocumentCollection:
        """Create a document collection from dense retrieval."""
        documents: list[Document]
        match dataset.provider:
            case "vespa":
                documents = await fetch_from_vespa_dense_retrieval(
                    queries=queries,
                    search_iteration=search_iteration,
                    fields=[*BASIC_FIELDS, "snippets"],
                    top_k=top_k,
                    dataset=dataset,
                    time_range=time_range,
                    venues=venues,
                    authors=authors,
                    corpus_ids=corpus_ids,
                    fields_of_study=fields_of_study,
                    # Honor the config-level vespa concurrency cap. Without
                    # this we'd fall back to the function default (10) and
                    # blow past the S2 1 req/sec cumulative rate limit.
                    vespa_concurrency=self._context.vespa_max_concurrency,
                    context=self._context,
                    inserted_before=inserted_before,
                )
        collection = self.from_docs(documents=documents)
        try:
            cache = self._context.cache
            await cache.put(collection.documents, collection.to_field_requirements(S2_FIELDS))
            collection = await collection.with_fields(["markdown"])
        except Exception as e:
            logging.exception(f"Failed to populate cache for dense retrieval documents (skipping): {e}")
        return collection

    async def from_openalex_search(
        self,
        query: str,
        limit: int,
        search_iteration: int = 1,
        time_range: ExtractedYearlyTimeRange | None = None,
        fields_of_study: list[str] | None = None,
    ) -> DocumentCollection:
        """Create a document collection from an OpenAlex `/works` search.

        OpenAlex docs are pre-populated with all standard fields (title,
        abstract, authors, etc.) at fetch time, so the dynamic-field
        loaders skip them. We deliberately don't call `with_fields`
        afterwards because OpenAlex corpus_ids are synthetic (`oa:Wxxx`)
        and would just no-op through the `from_s2` enrichment path.

        The cache is also skipped for the same reason - the cache layer
        keys on corpus_id assuming it round-trips to S2, which doesn't
        apply here.
        """
        documents = await fetch_from_openalex_search(
            queries=[query],
            search_iteration=search_iteration,
            top_k=limit,
            context=self._context,
            time_range=time_range,
            fields_of_study=fields_of_study,
        )
        return self.from_docs(documents=documents)
