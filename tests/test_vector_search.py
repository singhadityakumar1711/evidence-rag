"""Check search parameters and returned results without a running database."""

from unittest.mock import MagicMock

import pytest

from app.storage import vector_search


@pytest.fixture
def mock_connection(monkeypatch):
    connection = MagicMock()
    connection.__enter__.return_value = connection
    monkeypatch.setattr(vector_search, "get_connection", lambda: connection)
    return connection


def test_search_chunks_preserves_result_order_and_metadata(mock_connection):
    metadata = {"source": "sample://vectors", "chunk_index": 0}
    mock_connection.execute.return_value.fetchall.return_value = [
        ("doc-002:0000", "doc-002", "Vectors are stored in pgvector.", metadata, 0.1),
        ("doc-001:0000", "doc-001", "Documents are split into chunks.", {}, 0.4),
    ]

    results = vector_search.search_chunks(
        [1.0] + [0.0] * 383, top_k=5, model_name="test-model"
    )

    assert [result["chunk_id"] for result in results] == [
        "doc-002:0000",
        "doc-001:0000",
    ]
    assert [result["score"] for result in results] == pytest.approx([0.9, 0.6])
    assert results[0]["document_id"] == "doc-002"
    assert results[0]["text"] == "Vectors are stored in pgvector."
    assert results[0]["metadata"] == metadata


def test_search_chunks_returns_empty_list_when_no_chunks_match(mock_connection):
    mock_connection.execute.return_value.fetchall.return_value = []

    results = vector_search.search_chunks(
        [1.0] + [0.0] * 383, top_k=5, model_name="test-model"
    )

    assert results == []


def test_search_chunks_passes_vector_model_and_limit_as_parameters(mock_connection):
    mock_connection.execute.return_value.fetchall.return_value = []
    query_embedding = [1.0] + [0.0] * 383

    vector_search.search_chunks(query_embedding, top_k=3, model_name="test-model")

    mock_connection.execute.assert_called_once()
    _, parameters = mock_connection.execute.call_args.args
    assert parameters[0].to_list() == query_embedding
    assert parameters[1:] == ("test-model", 3)
