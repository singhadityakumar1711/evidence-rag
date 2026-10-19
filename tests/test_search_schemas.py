import pytest
from pydantic import ValidationError

from app.api.search_schemas import SearchRequest


def test_search_request_trims_query_and_defaults_to_five_results():
    request = SearchRequest(query="  How are embeddings stored?  ")

    assert request.query == "How are embeddings stored?"
    assert request.top_k == 5


@pytest.mark.parametrize("query", ["", "   ", "\t\n"])
def test_search_request_rejects_blank_queries(query: str):
    with pytest.raises(ValidationError):
        SearchRequest(query=query)


@pytest.mark.parametrize("top_k", [-1, 0, 21])
def test_search_request_rejects_top_k_outside_allowed_range(top_k: int):
    with pytest.raises(ValidationError):
        SearchRequest(query="How are embeddings stored?", top_k=top_k)


@pytest.mark.parametrize("top_k", [1, 20])
def test_search_request_accepts_top_k_at_range_boundaries(top_k: int):
    request = SearchRequest(query="How are embeddings stored?", top_k=top_k)

    assert request.top_k == top_k
