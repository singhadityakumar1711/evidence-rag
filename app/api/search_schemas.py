"""Request and response models for vector search."""

from pydantic import BaseModel, Field, field_validator


class SearchRequest(BaseModel):
    """A question and the maximum number of chunks to retrieve."""

    query: str = Field(min_length=1)
    top_k: int = Field(default=5, ge=1, le=20)

    @field_validator("query")
    @classmethod
    def validate_query(cls, query: str) -> str:
        query = query.strip()
        if not query:
            raise ValueError("query must contain non-whitespace text")
        return query


class SearchResult(BaseModel):
    """One retrieved chunk; higher cosine similarity scores rank first."""

    chunk_id: str
    document_id: str
    text: str
    score: float
    metadata: dict[str, str | int]


class SearchResponse(BaseModel):
    """The question and its ranked search results."""

    query: str
    results: list[SearchResult]
