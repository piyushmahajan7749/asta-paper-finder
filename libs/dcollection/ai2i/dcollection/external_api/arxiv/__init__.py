"""
Async arXiv client.

Free + keyless. arXiv's polite-pool rules: max 1 request per 3 seconds
(http://export.arxiv.org/help/api/user-manual). Our paper-finder side
keeps fetcher-level concurrency at 1 by default + adds a 4-second
spacing safety margin between calls within the same context.

API endpoint returns an Atom XML feed. We parse with stdlib
ElementTree (no extra dep) and map Atom <entry> elements to article
dicts compatible with the OpenAlex/PubMed shape.

API docs: https://info.arxiv.org/help/api/user-manual.html
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any

import httpx
from httpx import HTTPStatusError, NetworkError, TimeoutException

logger = logging.getLogger(__name__)


ARXIV_API_URL = "https://export.arxiv.org/api/query"
NS = {
    "atom": "http://www.w3.org/2005/Atom",
    "arxiv": "http://arxiv.org/schemas/atom",
}


class AsyncArxivClient:
    def __init__(self, timeout: int = 20) -> None:
        self.timeout = timeout
        logger.info("[arxiv] Client constructed")

    async def search(
        self,
        query: str,
        max_results: int = 25,
        time_range_start: int | None = None,
        time_range_end: int | None = None,
    ) -> list[dict[str, Any]]:
        """Search arXiv by keyword, return article dicts.

        Returns up to `max_results` entries sorted by arXiv's relevance
        score. Each dict has: arxiv_id, doi (if cross-referenced), title,
        abstract, authors, published_year, primary_category,
        publication_types (= ["preprint"] for everything from arXiv).
        """
        if not query or not query.strip():
            return []

        max_results = max(1, min(max_results, 100))

        # arXiv's `search_query` syntax accepts free-text via `all:` or
        # field-specific (ti:, abs:, etc.). We use `all:` so the query
        # matches across title + abstract + author fields.
        sq = f"all:{query.strip()}"
        if time_range_start or time_range_end:
            # arXiv's date format: YYYYMMDDHHMM. Anchor at the start of
            # the start-year and end of the end-year.
            start = f"{time_range_start}01010000" if time_range_start else "190001010000"
            end = f"{time_range_end}12312359" if time_range_end else "300001010000"
            sq = f"({sq}) AND submittedDate:[{start} TO {end}]"

        params: dict[str, Any] = {
            "search_query": sq,
            "max_results": max_results,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(ARXIV_API_URL, params=params)
                if r.status_code == 429:
                    logger.warning("[arxiv] rate-limited (429); returning empty.")
                    return []
                r.raise_for_status()
                xml_bytes = r.content
        except HTTPStatusError as e:
            logger.error(
                f"[arxiv] HTTP {e.response.status_code} for query={query[:80]!r}: "
                f"{e.response.text[:200]}"
            )
            return []
        except (NetworkError, TimeoutException) as e:
            logger.error(f"[arxiv] network/timeout for query={query[:80]!r}: {e}")
            return []

        return _parse_atom_feed(xml_bytes)


def _text(el: ET.Element | None) -> str:
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _parse_atom_feed(xml_bytes: bytes) -> list[dict[str, Any]]:
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        logger.error(f"[arxiv] Atom feed parse failed: {e}")
        return []

    results: list[dict[str, Any]] = []
    for entry in root.findall("atom:entry", NS):
        # arXiv ID is the LAST segment of <id> (the abs URL):
        #   http://arxiv.org/abs/2403.01234v1 → "2403.01234v1"
        # Strip the version suffix so de-dup across versions works.
        id_url = _text(entry.find("atom:id", NS))
        arxiv_id = id_url.rsplit("/", 1)[-1] if id_url else ""
        if "v" in arxiv_id and arxiv_id.split("v")[-1].isdigit():
            arxiv_id = arxiv_id.rsplit("v", 1)[0]
        if not arxiv_id:
            continue

        title = _text(entry.find("atom:title", NS))
        abstract = _text(entry.find("atom:summary", NS))

        authors: list[str] = []
        for a in entry.findall("atom:author", NS):
            name = _text(a.find("atom:name", NS))
            if name:
                authors.append(name)

        published = _text(entry.find("atom:published", NS))
        year: int | None = None
        if len(published) >= 4 and published[:4].isdigit():
            try:
                year = int(published[:4])
            except ValueError:
                year = None

        # Cross-referenced DOI (e.g. for accepted-in-journal papers).
        # Lives at <arxiv:doi> on the entry when present.
        doi = _text(entry.find("arxiv:doi", NS)).lower() or None

        primary_cat_el = entry.find("arxiv:primary_category", NS)
        primary_category = (
            primary_cat_el.attrib.get("term") if primary_cat_el is not None else None
        )

        # Try to extract the PDF link for the URL.
        pdf_url: str | None = None
        landing_url: str | None = None
        for link in entry.findall("atom:link", NS):
            href = link.attrib.get("href")
            if not href:
                continue
            if link.attrib.get("title") == "pdf" or link.attrib.get("type") == "application/pdf":
                pdf_url = href
            elif link.attrib.get("rel") == "alternate":
                landing_url = href

        results.append(
            {
                "arxiv_id": arxiv_id,
                "doi": doi,
                "title": title.replace("\n", " ").strip() or None,
                "abstract": abstract.replace("\n", " ").strip() or None,
                "authors": authors,
                "year": year,
                "primary_category": primary_category,
                "url": landing_url or pdf_url or f"https://arxiv.org/abs/{arxiv_id}",
                # Every arXiv entry IS a preprint by definition. The
                # downstream review-filter checks publicationTypes for
                # "review"/"meta-analysis"; "preprint" doesn't match
                # any of those so it's correctly flagged as primary
                # research.
                "publication_types": ["preprint"],
            }
        )

    return results
