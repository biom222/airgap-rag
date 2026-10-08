from uuid import UUID, uuid4

from airgap_rag.chunking.recursive import RecursiveCharacterChunker
from airgap_rag.documents.parsers import ParsedDocument, ParsedPage


def test_chunker_preserves_page_and_overlap() -> None:
    document_id = uuid4()
    parsed = ParsedDocument(pages=(ParsedPage(page=3, text="alpha beta gamma delta epsilon"),))

    chunks = RecursiveCharacterChunker(chunk_size=16, chunk_overlap=5).split(document_id, parsed)

    assert len(chunks) >= 2
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))
    assert all(chunk.page == 3 for chunk in chunks)
    assert all(len(chunk.text) <= 16 for chunk in chunks)
    assert chunks[0].text[-5:].strip() in chunks[1].text


def test_chunk_ids_are_deterministic() -> None:
    document_id = UUID("c7e58f7e-82bd-4f3b-87c8-2ce2af6bd269")
    parsed = ParsedDocument(pages=(ParsedPage(page=None, text="stable content"),))
    chunker = RecursiveCharacterChunker(chunk_size=100, chunk_overlap=10)

    first = chunker.split(document_id, parsed)
    second = chunker.split(document_id, parsed)

    assert first == second
    assert first[0].text_hash


def test_chunker_skips_empty_pages() -> None:
    parsed = ParsedDocument(
        pages=(ParsedPage(page=1, text="  "), ParsedPage(page=2, text="content"))
    )

    chunks = RecursiveCharacterChunker(100, 10).split(uuid4(), parsed)

    assert len(chunks) == 1
    assert chunks[0].page == 2
