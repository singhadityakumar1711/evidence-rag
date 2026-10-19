"""Check search HTTP behavior without loading a model or connecting to Postgres."""

from unittest.mock import MagicMock

import pytest
from fastapi.testclient import TestClient

from app import main


client = TestClient(main.app)


@pytest.fixture
def mock_retrieval(monkeypatch):
    retrieve = MagicMock(return_value=[])
    monkeypatch.setattr(main, "retrieve_chunks", retrieve)
    return retrieve


def test_search_returns_ranked_chunks_with_scores_and_metadata(mock_retrieval):
    results = [
        {
            "chunk_id": "doc-002:0000",
            "document_id": "doc-002",
            "text": "Vectors are stored in pgvector.",
            "score": 0.9,
            "metadata": {"source": "sample://vectors", "chunk_index": 0},
        },
        {
            "chunk_id": "doc-001:0000",
            "document_id": "doc-001",
            "text": "Documents are split into chunks.",
            "score": 0.6,
            "metadata": {"source": "sample://chunking", "chunk_index": 0},
        },
    ]
    mock_retrieval.return_value = results

    response = client.post(
        "/search", json={"query": "  Where are vectors stored?  ", "top_k": 2}
    )

    assert response.status_code == 200
    assert response.json() == {
        "query": "Where are vectors stored?",
        "results": results,
    }
    mock_retrieval.assert_called_once_with("Where are vectors stored?", top_k=2)


def test_search_defaults_to_five_and_returns_empty_results(mock_retrieval):
    response = client.post("/search", json={"query": "Where are vectors stored?"})

    assert response.status_code == 200
    assert response.json() == {"query": "Where are vectors stored?", "results": []}
    mock_retrieval.assert_called_once_with("Where are vectors stored?", top_k=5)


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"query": ""},
        {"query": "   "},
        {"query": "Where are vectors stored?", "top_k": 0},
        {"query": "Where are vectors stored?", "top_k": 21},
    ],
)
def test_search_rejects_invalid_requests_before_retrieval(mock_retrieval, payload):
    response = client.post("/search", json=payload)

    assert response.status_code == 422
    mock_retrieval.assert_not_called()
