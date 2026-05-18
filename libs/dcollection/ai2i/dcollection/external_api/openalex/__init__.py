"""
Async OpenAlex client.

OpenAlex (https://openalex.org) is a free, open scholarly index covering
~250M works. It requires no API key - opt into the "polite pool"
(higher rate limits + priority) by passing your contact email via the
`mailto=` query param.

We added OpenAlex as a parallel arm to S2 in fast_broad_search so the
service stays useful when S2's key/quota is degraded (the 2026-05-18
incident: expired S2_API_KEY → every dense + S2 paper-search call
403'd, lit-scout returned zero papers across the board).

API docs: https://docs.openalex.org/api-entities/works
"""

from __future__ import annotations

import logging
from typing import Any, Sequence

import httpx
from httpx import HTTPStatusError, NetworkError, TimeoutException

logger = logging.getLogger(__name__)


# OpenAlex's `select=` param trims response payload. Each work is
# ~50 fields by default and a 25-result page can balloon past 1 MB,
# which is wasteful when we only render ~10 fields. Tight `select`
# list also makes the response parse much cheaper.
DEFAULT_WORK_FIELDS: tuple[str, ...] = (
    "id",
    "doi",
    "title",
    "publication_year",
    "publication_date",
    "type",
    "cited_by_count",
    "authorships",
    "primary_location",
    "abstract_inverted_index",
    "open_access",
    "concepts",
    "ids",
)

# Hard cap. OpenAlex allows `per_page` up to 200, but past ~50 the
# response is huge and most ranks past that are noise. Keep it tight.
MAX_PER_PAGE: int = 50


class AsyncOpenAlexClient:
    """Minimal async OpenAlex client.

    Just `/works?search=...` for now - that's all `fast_broad_search`
    needs. Other endpoints (`/authors`, `/concepts`, single-work
    fetch by ID) are easy to add later if a downstream agent needs
    them.

    Auth model: `mailto=` opt-in only. No bearer token. If `mailto`
    is empty, requests still work but land in the common pool with
    aggressive rate limiting; we log a warning at construction.
    """

    def __init__(
        self,
        mailto: str | None = None,
        timeout: int = 15,
        base_url: str = "https://api.openalex.org",
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.mailto = mailto.strip() if isinstance(mailto, str) and mailto.strip() else None
        if not self.mailto:
            logger.warning(
                "[openalex] OPENALEX_MAILTO not set - falling back to common pool with aggressive "
                "rate limiting. Set the env var to your contact email to opt into the polite pool "
                "(higher rate limits + priority). Docs: "
                "https://docs.openalex.org/how-to-use-the-api/api-overview#user-agent"
            )
        else:
            logger.info(f"[openalex] Polite-pool client constructed (mailto={_redact_email(self.mailto)})")

    @property
    def _user_agent(self) -> str:
        return f"asta-paper-finder ({self.mailto or 'noreply@example.com'})"

    async def search_works(
        self,
        query: str,
        per_page: int = 25,
        time_range_start: int | None = None,
        time_range_end: int | None = None,
        fields_of_study: Sequence[str] | None = None,
        fields: Sequence[str] = DEFAULT_WORK_FIELDS,
    ) -> list[dict[str, Any]]:
        """Search OpenAlex works by keyword.

        Returns the raw `results` array from the OpenAlex response.
        Each entry is a dict; downstream fetcher code maps these into
        PaperFinderDocument instances.

        Errors are logged and re-raised so the caller's gather()
        return_exceptions=True path treats them as failed arms.
        """
        if not query or not query.strip():
            return []

        per_page = max(1, min(per_page, MAX_PER_PAGE))
        params: dict[str, Any] = {
            "search": query.strip(),
            "per_page": per_page,
            "select": ",".join(fields),
        }
        if self.mailto:
            params["mailto"] = self.mailto

        # OpenAlex filter expressions are comma-separated `key:value` pairs
        # passed as a single `filter=` param. Build it up incrementally.
        filters: list[str] = []
        if time_range_start and time_range_end and time_range_start <= time_range_end:
            filters.append(f"from_publication_date:{time_range_start}-01-01")
            filters.append(f"to_publication_date:{time_range_end}-12-31")
        elif time_range_start:
            filters.append(f"from_publication_date:{time_range_start}-01-01")
        elif time_range_end:
            filters.append(f"to_publication_date:{time_range_end}-12-31")
        if fields_of_study:
            # OpenAlex maps fields-of-study via `concepts.id` filters.
            # We don't have stable concept IDs from upstream callers,
            # so pass them as `concepts.display_name.search:` which
            # matches by name. Best-effort - extra signal, not a hard
            # filter.
            for fos in fields_of_study:
                if fos:
                    filters.append(f"concepts.display_name.search:{fos}")
        if filters:
            params["filter"] = ",".join(filters)

        headers = {"User-Agent": self._user_agent}

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                response = await client.get(
                    f"{self.base_url}/works",
                    params=params,
                    headers=headers,
                )
                if response.status_code == 429:
                    logger.warning(
                        "[openalex] Rate limited (429). Caller should back off; returning empty result."
                    )
                    return []
                response.raise_for_status()
                payload = response.json()
        except HTTPStatusError as e:
            logger.error(
                f"[openalex] HTTP {e.response.status_code} on /works for query={query[:80]!r}: "
                f"{e.response.text[:200]}"
            )
            raise
        except (NetworkError, TimeoutException) as e:
            logger.error(f"[openalex] Network/timeout error on /works for query={query[:80]!r}: {e}")
            raise

        results = payload.get("results")
        if not isinstance(results, list):
            logger.warning(f"[openalex] Unexpected response shape for query={query[:80]!r}: no 'results' array")
            return []
        return results


def reconstruct_abstract_from_inverted_index(inverted: dict[str, Any] | None) -> str:
    """Convert OpenAlex `abstract_inverted_index` back to linear text.

    OpenAlex stores abstracts as `{word: [pos1, pos2, ...]}` to avoid
    redistributing copyrighted long-form text. To recover prose: flatten
    to `[(pos, word), ...]`, sort by position, join with spaces. Some
    words appear at multiple positions (one entry per position), which
    is handled naturally by the flatten step.

    Returns "" when the work has no abstract (some preprints + grey
    literature). Callers should treat that as "abstract unavailable".
    """
    if not inverted or not isinstance(inverted, dict):
        return ""
    tokens: list[tuple[int, str]] = []
    for word, positions in inverted.items():
        if not isinstance(positions, list):
            continue
        for pos in positions:
            if isinstance(pos, int):
                tokens.append((pos, word))
    tokens.sort(key=lambda t: t[0])
    return " ".join(word for _pos, word in tokens)


def _redact_email(email: str) -> str:
    """Show `a***@example.com` style hint without exposing the local-part."""
    if "@" not in email:
        return "***"
    local, domain = email.split("@", 1)
    if not local:
        return f"***@{domain}"
    return f"{local[0]}***@{domain}"
