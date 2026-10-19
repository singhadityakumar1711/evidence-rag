"""Turn a text question into ranked document chunks."""

from app.processing.embeddings import (
    EMBEDDING_MODEL_NAME,
    embed_texts,
    get_embedding_model,
)
from app.storage.vector_search import search_chunks


def retrieve_chunks(query: str, *, top_k: int = 5) -> list[dict]:
    """Embed a validated question and retrieve its closest stored chunks."""
    model = get_embedding_model()
    # embed_texts returns one vector per input text; we supplied one question.
    query_embedding = embed_texts(model, [query])[0]

    return search_chunks(
        query_embedding,
        top_k=top_k,
        model_name=EMBEDDING_MODEL_NAME,
    )
