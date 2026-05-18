"""
Reproduce the OpenAlex field-wipe bug locally.

Walk the doc through the same pipeline the deployed agent does:
  1. Fetcher builds the doc (title/abstract/url set).
  2. Factory wraps into a collection.
  3. `with_fields(["markdown"])` triggers the full loader chain
     (load required_fields title/authors/year/abstract via from_s2,
     then load_markdown).
  4. Print title at each stage to see WHERE it gets wiped.

Run with the project venv:
    .venv/bin/python scripts/test_openalex_fetcher.py
"""

from __future__ import annotations

import asyncio
import sys


def _print_doc_state(label: str, doc) -> None:
    """Compact one-line summary of doc state at each pipeline stage."""
    title = doc.title
    title_str = (title[:60] + "...") if isinstance(title, str) else repr(title)
    abstract_present = isinstance(doc.abstract, str) and doc.abstract != ""
    url_present = isinstance(doc.url, str) and doc.url != ""
    print(
        f"  [{label}] "
        f"title={title_str!r}  "
        f"url_set={url_present}  "
        f"abstract_set={abstract_present}  "
        f"is_loaded(title)={doc.is_loaded('title')}  "
        f"loaded_fields_count={len(doc._loaded_fields)}"
    )


async def main() -> int:
    from ai2i.dcollection.external_api.openalex import AsyncOpenAlexClient
    from ai2i.dcollection.factory import DocumentCollectionFactory
    from ai2i.dcollection.fetchers.openalex import _doc_from_openalex_work

    # 1) Get one raw OpenAlex result + build a doc directly
    client = AsyncOpenAlexClient(mailto="test@example.com")
    results = await client.search_works(
        "lithium dendrite formation LLZO solid electrolyte",
        per_page=1,
    )
    print(f"OpenAlex returned {len(results)} raw result(s)\n")
    if not results:
        print("Nothing to test")
        return 1

    raw = results[0]
    print(f"raw OpenAlex title = {(raw.get('title') or '')[:80]!r}\n")

    # 2) Build a doc via fetcher
    doc = _doc_from_openalex_work(
        work=raw, query="test", rank=1, search_iteration=1
    )
    if doc is None:
        print("Fetcher returned None - aborting")
        return 1

    _print_doc_state("after _doc_from_openalex_work", doc)

    # 3) Wrap into a collection via the factory (this triggers fuse via from_docs)
    factory = DocumentCollectionFactory(
        s2_api_key=None,  # We have no S2 key locally; that's fine
        openalex_mailto="test@example.com",
    )
    collection = factory.from_docs(documents=[doc])
    doc_after_from_docs = collection.documents[0]
    _print_doc_state("after factory.from_docs", doc_after_from_docs)

    # 4) Call with_fields(["markdown"]) - this is what _run_relevance_judgement and
    # the response-build path do. Walks the full loader chain.
    print("\n>>> Calling collection.with_fields(['markdown'])...")
    try:
        collection_after = await collection.with_fields(["markdown"])
        doc_after_with_fields = collection_after.documents[0]
        _print_doc_state("after with_fields([markdown])", doc_after_with_fields)
        print(
            f"  markdown[:120] = {((doc_after_with_fields.markdown or '')[:120])!r}"
        )
    except Exception as e:
        print(f"  with_fields(['markdown']) raised: {type(e).__name__}: {e}")

    # 5) Verify the ORIGINAL doc in the original collection too (Python mutability)
    print()
    _print_doc_state("original `doc` (post-call)", doc)
    _print_doc_state(
        "collection.documents[0] (post-call)", collection.documents[0]
    )

    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
