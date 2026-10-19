"""Check query embedding and search wiring without model downloads or a database."""

from unittest.mock import MagicMock

import pytest

from app.processing import retrieval
from app.processing.embeddings import EMBEDDING_MODEL_NAME


@pytest.fixture
def mock_model_and_search(monkeypatch):
    model = MagicMock()
    model.encode.return_value.tolist.return_value = [[1.0] + [0.0] * 383]
    search = MagicMock(return_value=[])

    monkeypatch.setattr(retrieval, "get_embedding_model", lambda: model)
    monkeypatch.setattr(retrieval, "search_chunks", search)
    return model, search


def test_retrieve_chunks_searches_using_the_question_embedding(mock_model_and_search):
    model, search = mock_model_and_search
    expected_results = [
        {
            "chunk_id": "doc-001:0000",
            "document_id": "doc-001",
            "text": "Vectors are stored in pgvector.",
            "score": 0.9,
            "metadata": {"source": "sample://vectors"},
        }
    ]
    search.return_value = expected_results

    results = retrieval.retrieve_chunks("Where are vectors stored?", top_k=3)

    model.encode.assert_called_once()
    assert model.encode.call_args.args[0] == ["Where are vectors stored?"]
    search.assert_called_once_with(
        [1.0] + [0.0] * 383,
        top_k=3,
        model_name=EMBEDDING_MODEL_NAME,
    )
    assert results == expected_results


def test_retrieve_chunks_defaults_to_five_and_handles_no_matches(mock_model_and_search):
    _, search = mock_model_and_search

    results = retrieval.retrieve_chunks("Where are vectors stored?")

    assert search.call_args.kwargs["top_k"] == 5
    assert results == []
