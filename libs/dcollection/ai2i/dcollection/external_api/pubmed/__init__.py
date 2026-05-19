"""
Async PubMed client (NCBI eutils).

Free + keyless: 3 requests/second base rate. With NCBI_API_KEY env var
(register at https://www.ncbi.nlm.nih.gov/account/) the limit jumps to
10 req/s — recommended for production.

Two-stage retrieval flow:
  1. `esearch` — query → list of PMIDs (JSON, fast).
  2. `efetch`  — PMIDs → full article XML (one call for the whole batch).

Why XML over esummary's JSON: esummary's payload is missing the
abstract text. We always want abstracts for the LLM-relevance step, so
we use efetch + parse the PubmedArticle XML tree.

API docs: https://www.ncbi.nlm.nih.gov/books/NBK25500/
"""

from __future__ import annotations

import logging
import xml.etree.ElementTree as ET
from typing import Any, Sequence

import httpx
from httpx import HTTPStatusError, NetworkError, TimeoutException

logger = logging.getLogger(__name__)


ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# NCBI is strict about identifying clients. They REQUIRE a `tool` +
# `email` param on every call so they can contact us if a script is
# misbehaving. We pass our project name + a contact pulled from
# PUBMED_CONTACT_EMAIL (or fall back to a generic noreply so requests
# never go un-tagged).
DEFAULT_TOOL = "asta-paper-finder"


class AsyncPubMedClient:
    def __init__(
        self,
        api_key: str | None = None,
        contact_email: str | None = None,
        timeout: int = 20,
    ) -> None:
        self.api_key = api_key.strip() if isinstance(api_key, str) and api_key.strip() else None
        self.contact_email = (
            contact_email.strip()
            if isinstance(contact_email, str) and contact_email.strip()
            else "noreply@example.com"
        )
        self.timeout = timeout
        if self.api_key:
            logger.info("[pubmed] Client constructed WITH NCBI_API_KEY (10 req/s ceiling)")
        else:
            logger.info("[pubmed] Client constructed WITHOUT NCBI_API_KEY (3 req/s base rate)")

    async def search(
        self,
        query: str,
        max_results: int = 25,
        time_range_start: int | None = None,
        time_range_end: int | None = None,
    ) -> list[dict[str, Any]]:
        """Run esearch + efetch and return a list of article dicts.

        Each entry has: pmid, doi (or None), title, abstract, authors
        (list[str]), journal, year, publication_types (list[str]).
        Order matches PubMed's relevance ranking when sort=relevance.
        """
        if not query or not query.strip():
            return []

        max_results = max(1, min(max_results, 100))

        # ── esearch ──────────────────────────────────────────────────────
        esearch_params: dict[str, Any] = {
            "db": "pubmed",
            "term": query.strip(),
            "retmax": max_results,
            "retmode": "json",
            "sort": "relevance",
            "tool": DEFAULT_TOOL,
            "email": self.contact_email,
        }
        if self.api_key:
            esearch_params["api_key"] = self.api_key
        if time_range_start or time_range_end:
            # NCBI uses YYYY/MM/DD for date filters; we have year only.
            start = f"{time_range_start}/01/01" if time_range_start else "1900/01/01"
            end = f"{time_range_end}/12/31" if time_range_end else "3000/12/31"
            esearch_params["mindate"] = start
            esearch_params["maxdate"] = end
            esearch_params["datetype"] = "pdat"

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(ESEARCH_URL, params=esearch_params)
                if r.status_code == 429:
                    logger.warning("[pubmed] esearch rate-limited (429); returning empty.")
                    return []
                r.raise_for_status()
                esearch = r.json()
        except HTTPStatusError as e:
            logger.error(
                f"[pubmed] esearch HTTP {e.response.status_code} for query={query[:80]!r}: "
                f"{e.response.text[:200]}"
            )
            return []
        except (NetworkError, TimeoutException) as e:
            logger.error(f"[pubmed] esearch network/timeout for query={query[:80]!r}: {e}")
            return []

        pmids: list[str] = (
            esearch.get("esearchresult", {}).get("idlist", []) if isinstance(esearch, dict) else []
        )
        if not pmids:
            logger.info(f"[pubmed] esearch returned 0 PMIDs for query={query[:80]!r}")
            return []

        # ── efetch (XML) ─────────────────────────────────────────────────
        efetch_params: dict[str, Any] = {
            "db": "pubmed",
            "id": ",".join(pmids),
            "rettype": "abstract",
            "retmode": "xml",
            "tool": DEFAULT_TOOL,
            "email": self.contact_email,
        }
        if self.api_key:
            efetch_params["api_key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self.timeout) as client:
                r = await client.get(EFETCH_URL, params=efetch_params)
                if r.status_code == 429:
                    logger.warning("[pubmed] efetch rate-limited (429); returning empty.")
                    return []
                r.raise_for_status()
                xml_bytes = r.content
        except HTTPStatusError as e:
            logger.error(f"[pubmed] efetch HTTP {e.response.status_code}: {e.response.text[:200]}")
            return []
        except (NetworkError, TimeoutException) as e:
            logger.error(f"[pubmed] efetch network/timeout: {e}")
            return []

        return _parse_pubmed_xml(xml_bytes, pmids)


def _text(el: ET.Element | None) -> str:
    """Get text from an Element, joining nested children (for AbstractText
    with bolded sections)."""
    if el is None:
        return ""
    return "".join(el.itertext()).strip()


def _parse_pubmed_xml(xml_bytes: bytes, pmid_order: Sequence[str]) -> list[dict[str, Any]]:
    """Parse the efetch PubmedArticleSet XML into a list of article dicts.

    Preserves the order from `pmid_order` (PubMed's relevance ranking
    from esearch). Articles missing from efetch are silently skipped.
    """
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError as e:
        logger.error(f"[pubmed] XML parse failed: {e}")
        return []

    # Index by PMID so we can stitch ordered results below.
    by_pmid: dict[str, dict[str, Any]] = {}
    for article_el in root.findall(".//PubmedArticle"):
        pmid_el = article_el.find(".//MedlineCitation/PMID")
        pmid = _text(pmid_el)
        if not pmid:
            continue

        title = _text(article_el.find(".//Article/ArticleTitle"))

        # AbstractText elements may have a Label attr (Background /
        # Methods / Results / Conclusions). Join them into one prose
        # block separated by section headers when present.
        abstract_parts: list[str] = []
        for at in article_el.findall(".//Abstract/AbstractText"):
            label = at.attrib.get("Label", "").strip()
            body = _text(at)
            if not body:
                continue
            abstract_parts.append(f"{label}: {body}" if label else body)
        abstract = " ".join(abstract_parts)

        authors: list[str] = []
        for a in article_el.findall(".//AuthorList/Author"):
            last = _text(a.find("LastName"))
            fore = _text(a.find("ForeName"))
            if last and fore:
                authors.append(f"{fore} {last}")
            elif last:
                authors.append(last)
            else:
                collective = _text(a.find("CollectiveName"))
                if collective:
                    authors.append(collective)

        journal = _text(article_el.find(".//Journal/Title"))

        # Publication year: prefer the Article PubDate Year, else
        # MedlineDate "2024 Jul-Aug" → grab the first 4 digits.
        year_str = _text(article_el.find(".//Article/Journal/JournalIssue/PubDate/Year"))
        if not year_str:
            mdate = _text(article_el.find(".//Article/Journal/JournalIssue/PubDate/MedlineDate"))
            for tok in mdate.split():
                if len(tok) >= 4 and tok[:4].isdigit():
                    year_str = tok[:4]
                    break
        year: int | None = None
        if year_str.isdigit():
            try:
                year = int(year_str)
            except ValueError:
                year = None

        # DOI lives in ArticleIdList alongside PMID/PMC. Pick the one
        # with IdType="doi".
        doi: str | None = None
        for id_el in article_el.findall(".//PubmedData/ArticleIdList/ArticleId"):
            if id_el.attrib.get("IdType", "").lower() == "doi":
                doi = _text(id_el).lower() or None
                if doi:
                    break

        # PublicationType is how we detect reviews / meta-analyses
        # without a title-heuristic. Always populated for PubMed.
        pub_types = [
            _text(pt) for pt in article_el.findall(".//Article/PublicationTypeList/PublicationType")
        ]
        pub_types = [pt for pt in pub_types if pt]

        by_pmid[pmid] = {
            "pmid": pmid,
            "doi": doi,
            "title": title or None,
            "abstract": abstract or None,
            "authors": authors,
            "journal": journal or None,
            "year": year,
            "publication_types": pub_types,
        }

    # Preserve esearch's relevance order.
    return [by_pmid[p] for p in pmid_order if p in by_pmid]
