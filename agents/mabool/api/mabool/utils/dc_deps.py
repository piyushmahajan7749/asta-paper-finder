import logging

from ai2i.config import ConfigValue, configurable
from ai2i.dcollection import DocumentCollection, DocumentCollectionFactory
from ai2i.di import DI, create_module

from mabool.data_model.config import cfg_schema

logger = logging.getLogger(__name__)

dc_module = create_module("DocumentCollection")


def _describe_s2_key_state(s2_api_key: str | None) -> str:
    """Build a single-line, non-leaking summary of the S2 key state.

    We log this at every factory construction so an operator can
    confirm from the Azure log stream that the app actually saw the
    `S2_API_KEY` value from Application settings. The value itself is
    never logged - only its length and a hash-like fingerprint of the
    first/last 2 chars, which is enough to tell "wrong key" apart from
    "no key" without exposing the secret.
    """
    if not s2_api_key:
        return "MISSING (factory will run unauthenticated; expect 403 from snippet/search)"
    stripped = s2_api_key.strip()
    if not stripped:
        return "BLANK after strip (whitespace only)"
    if len(stripped) < 8:
        return f"PRESENT but suspiciously short (length={len(stripped)})"
    fingerprint = f"{stripped[:2]}…{stripped[-2:]}"
    return f"PRESENT (length={len(stripped)}, fingerprint={fingerprint})"


def _describe_openalex_mailto(mailto: str | None) -> str:
    """Same shape as the S2 describer: never log the full address."""
    if not mailto:
        return "MISSING (OpenAlex will use the common pool with aggressive rate limits)"
    stripped = mailto.strip()
    if not stripped or "@" not in stripped:
        return "INVALID (must be a contact email)"
    local, domain = stripped.split("@", 1)
    initial = local[0] if local else "?"
    return f"PRESENT ({initial}***@{domain})"


def _describe_key_present(key_name: str, value: str | None) -> str:
    """Short, non-leaking summary of an API-key env var. Same idea as
    `_describe_s2_key_state` - log just enough fingerprint to confirm
    the right value is present, never the secret itself."""
    if not value:
        return f"{key_name}: MISSING (arm will short-circuit to [])"
    s = value.strip()
    if not s:
        return f"{key_name}: BLANK"
    if len(s) < 8:
        return f"{key_name}: PRESENT but short (length={len(s)})"
    return f"{key_name}: PRESENT (length={len(s)}, fingerprint={s[:2]}…{s[-2:]})"


@dc_module.provides(scope="singleton")
async def round_doc_collection_factory(
    s2_api_key: str = DI.config(cfg_schema.s2_api_key),
    s2_api_timeout: int = DI.config(cfg_schema.s2_api.timeout),
    s2_max_concurrency: int = DI.config(cfg_schema.s2_api.concurrency),
    vespa_max_concurrency: int = DI.config(cfg_schema.vespa.concurrency),
    cache_ttl: int = DI.config(cfg_schema.cache.ttl),
    cache_is_enabled: bool = DI.config(cfg_schema.cache.enabled),
    force_deterministic: bool = DI.config(cfg_schema.force_deterministic),
    openalex_mailto: str | None = DI.config(cfg_schema.openalex.mailto, default=None),
    openalex_timeout: int = DI.config(cfg_schema.openalex.timeout, default=15),
    pubmed_api_key: str | None = DI.config(cfg_schema.pubmed.api_key, default=None),
    pubmed_contact_email: str | None = DI.config(cfg_schema.pubmed.contact_email, default=None),
    arxiv_timeout: int = DI.config(cfg_schema.arxiv.timeout, default=20),
    scholar_api_key: str | None = DI.config(cfg_schema.scholar.api_key, default=None),
    tavily_api_key: str | None = DI.config(cfg_schema.tavily.api_key, default=None),
) -> DocumentCollectionFactory:
    logger.info(f"[dc_deps] Building round DocumentCollectionFactory; S2_API_KEY: {_describe_s2_key_state(s2_api_key)}")
    logger.info(
        f"[dc_deps] S2 concurrency caps: s2_api={s2_max_concurrency}, vespa={vespa_max_concurrency} "
        f"(should both be 1 for the 1 req/s cumulative S2 rate limit)"
    )
    logger.info(f"[dc_deps] OpenAlex mailto: {_describe_openalex_mailto(openalex_mailto)}")
    logger.info(f"[dc_deps] {_describe_key_present('PUBMED_API_KEY', pubmed_api_key)}")
    logger.info(f"[dc_deps] {_describe_key_present('SERPAPI_API_KEY (scholar)', scholar_api_key)}")
    logger.info(f"[dc_deps] {_describe_key_present('TAVILY_API_KEY', tavily_api_key)}")
    dc_factory = DocumentCollectionFactory(
        s2_api_key=s2_api_key,
        s2_api_timeout=s2_api_timeout,
        s2_max_concurrency=s2_max_concurrency,
        vespa_max_concurrency=vespa_max_concurrency,
        cache_ttl=cache_ttl,
        cache_is_enabled=cache_is_enabled,
        force_deterministic=force_deterministic,
        openalex_mailto=openalex_mailto,
        openalex_timeout=openalex_timeout,
        pubmed_api_key=pubmed_api_key,
        pubmed_contact_email=pubmed_contact_email,
        arxiv_timeout=arxiv_timeout,
        scholar_api_key=scholar_api_key,
        tavily_api_key=tavily_api_key,
    )
    return dc_factory


@configurable
def detached_doc_collection_factory(
    s2_api_key: str = ConfigValue(cfg_schema.s2_api_key),
    s2_api_timeout: int = ConfigValue(cfg_schema.s2_api.timeout),
    s2_max_concurrency: int = ConfigValue(cfg_schema.s2_api.concurrency),
    vespa_max_concurrency: int = ConfigValue(cfg_schema.vespa.concurrency),
    cache_ttl: int = ConfigValue(cfg_schema.cache.ttl),
    cache_is_enabled: bool = ConfigValue(cfg_schema.cache.enabled),
    force_deterministic: bool = ConfigValue(cfg_schema.force_deterministic),
    openalex_mailto: str | None = ConfigValue(cfg_schema.openalex.mailto, default=None),
    openalex_timeout: int = ConfigValue(cfg_schema.openalex.timeout, default=15),
    pubmed_api_key: str | None = ConfigValue(cfg_schema.pubmed.api_key, default=None),
    pubmed_contact_email: str | None = ConfigValue(cfg_schema.pubmed.contact_email, default=None),
    arxiv_timeout: int = ConfigValue(cfg_schema.arxiv.timeout, default=20),
    scholar_api_key: str | None = ConfigValue(cfg_schema.scholar.api_key, default=None),
    tavily_api_key: str | None = ConfigValue(cfg_schema.tavily.api_key, default=None),
) -> DocumentCollectionFactory:
    dc_factory = DocumentCollectionFactory(
        s2_api_key=s2_api_key,
        s2_api_timeout=s2_api_timeout,
        s2_max_concurrency=s2_max_concurrency,
        vespa_max_concurrency=vespa_max_concurrency,
        cache_ttl=cache_ttl,
        cache_is_enabled=cache_is_enabled,
        force_deterministic=force_deterministic,
        openalex_mailto=openalex_mailto,
        openalex_timeout=openalex_timeout,
        pubmed_api_key=pubmed_api_key,
        pubmed_contact_email=pubmed_contact_email,
        arxiv_timeout=arxiv_timeout,
        scholar_api_key=scholar_api_key,
        tavily_api_key=tavily_api_key,
    )
    return dc_factory


@dc_module.provides(scope="singleton")
async def empty_doc_collection(
    dmf: DocumentCollectionFactory = DI.requires(round_doc_collection_factory),
) -> DocumentCollection:
    return dmf.empty()
