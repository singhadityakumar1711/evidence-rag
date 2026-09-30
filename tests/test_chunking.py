import pytest

from app.chunking import chunk_document
from app.documents import ingest_text


def make_document(text: str):
    return ingest_text(
        document_id="doc-001",
        source="https://example.com/guide",
        title="RAG Guide",
        text=text,
        document_type="html",
    )


def test_fixed_size_chunking_splits_text_at_the_configured_size():
    chunks = chunk_document(
        make_document("abcdefghij"), strategy="fixed_size", chunk_size=4
    )

    assert [chunk.text for chunk in chunks] == ["abcd", "efgh", "ij"]
    assert [chunk.chunk_id for chunk in chunks] == ["doc-001:0000", "doc-001:0001", "doc-001:0002"]


def test_fixed_size_chunking_adds_character_overlap():
    chunks = chunk_document(
        make_document("abcdefghij"),
        strategy="fixed_size",
        chunk_size=4,
        overlap=2,
    )

    assert [chunk.text for chunk in chunks] == ["abcd", "cdef", "efgh", "ghij"]


def test_sentence_chunking_keeps_sentences_together():
    chunks = chunk_document(
        make_document("One. Two. Three."), strategy="sentence", chunk_size=10
    )

    assert [chunk.text for chunk in chunks] == ["One. Two.", "Three."]


def test_sentence_chunking_repeats_a_sentence_for_overlap():
    chunks = chunk_document(
        make_document("One. Two. Three."),
        strategy="sentence",
        chunk_size=12,
        overlap=5,
    )

    assert [chunk.text for chunk in chunks] == ["One. Two.", "Two. Three."]


def test_chunk_metadata_preserves_document_provenance_and_offsets():
    document = make_document("Alpha. Beta.")
    chunk = chunk_document(document, strategy="sentence", chunk_size=20)[0]

    assert chunk.document_id == document.document_id
    assert chunk.metadata == {
        "source": "https://example.com/guide",
        "title": "RAG Guide",
        "document_type": "html",
        "chunk_index": 0,
        "start_char": 0,
        "end_char": 12,
        "strategy": "sentence",
    }


@pytest.mark.parametrize(
    ("chunk_size", "overlap"),
    [(0, 0), (10, -1), (10, 10)],
)
def test_chunking_rejects_invalid_sizes(chunk_size: int, overlap: int):
    with pytest.raises(ValueError):
        chunk_document(
            make_document("A document."),
            strategy="fixed_size",
            chunk_size=chunk_size,
            overlap=overlap,
        )
