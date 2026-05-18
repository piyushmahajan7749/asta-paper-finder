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


@dc_module.provides(scope="singleton")
async def round_doc_collection_factory(
    s2_api_key: str = DI.config(cfg_schema.s2_api_key),
    s2_api_timeout: int = DI.config(cfg_schema.s2_api.timeout),
    cache_ttl: int = DI.config(cfg_schema.cache.ttl),
    cache_is_enabled: bool = DI.config(cfg_schema.cache.enabled),
    force_deterministic: bool = DI.config(cfg_schema.force_deterministic),
) -> DocumentCollectionFactory:
    logger.info(f"[dc_deps] Building round DocumentCollectionFactory; S2_API_KEY: {_describe_s2_key_state(s2_api_key)}")
    dc_factory = DocumentCollectionFactory(
        s2_api_key=s2_api_key,
        s2_api_timeout=s2_api_timeout,
        cache_ttl=cache_ttl,
        cache_is_enabled=cache_is_enabled,
        force_deterministic=force_deterministic,
    )
    return dc_factory


@configurable
def detached_doc_collection_factory(
    s2_api_key: str = ConfigValue(cfg_schema.s2_api_key),
    s2_api_timeout: int = ConfigValue(cfg_schema.s2_api.timeout),
    cache_ttl: int = ConfigValue(cfg_schema.cache.ttl),
    cache_is_enabled: bool = ConfigValue(cfg_schema.cache.enabled),
    force_deterministic: bool = ConfigValue(cfg_schema.force_deterministic),
) -> DocumentCollectionFactory:
    dc_factory = DocumentCollectionFactory(
        s2_api_key=s2_api_key,
        s2_api_timeout=s2_api_timeout,
        cache_ttl=cache_ttl,
        cache_is_enabled=cache_is_enabled,
        force_deterministic=force_deterministic,
    )
    return dc_factory


@dc_module.provides(scope="singleton")
async def empty_doc_collection(
    dmf: DocumentCollectionFactory = DI.requires(round_doc_collection_factory),
) -> DocumentCollection:
    return dmf.empty()
