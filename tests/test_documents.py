from app.documents import Document, ingest_text


def test_ingest_text():
    document = ingest_text(
        document_id="doc-001",
        source="example.md",
        title="Test Document",
        text="Hello RAG.",
        document_type="markdown",
    )

    assert isinstance(document, Document)
    assert document.document_id == "doc-001"
    assert document.text == "Hello RAG."


def test_ingest_text_cleans_whitespace():
    document = ingest_text(
        document_id="doc-002",
        source="example.md",
        title="Whitespace Test",
        text="   Hello   world.\n\nThis is RAG.   ",
        document_type="markdown",
    )

    assert document.text == "Hello world. This is RAG."