import pytest

from app.paged_source_reader_v3 import catalog_pages, consolidation_catalog
from tests.test_routed_source_reader import sources


def test_all_original_chunks_survive_pagination_in_order_without_implicit_cutoff():
    _, docs, chunks, directory = sources()
    pages = catalog_pages(directory, directory.document_ids, docs, chunks, max_bytes=1100)
    assert len(pages) > 1
    expected = tuple(sorted(chunks, key=lambda k: chunks[k].start_char))
    assert tuple(k for p in pages for k in p.source_ids) == expected
    assert all(len(p.message.encode()) <= 1100 for p in pages)


def test_page_cap_returns_error_instead_of_silent_partial_document_reading():
    _, docs, chunks, directory = sources()
    with pytest.raises(ValueError, match="NO_TRUNCATION"):
        catalog_pages(directory, directory.document_ids, docs, chunks, max_bytes=1100, max_pages=1)


def test_final_reading_catalog_contains_only_actual_per_page_selections():
    _, docs, chunks, directory = sources()
    pages = catalog_pages(directory, directory.document_ids, docs, chunks, max_bytes=1100)
    selections = tuple(p.source_ids[:1] for p in pages)
    final = consolidation_catalog(directory, selections, pages, docs, chunks)
    assert final.source_ids == tuple(k for row in selections for k in row)
    with pytest.raises(ValueError, match="UNSUPPLIED_SELECTION"):
        consolidation_catalog(directory, (("invented-source",), *selections[1:]), pages, docs, chunks)
