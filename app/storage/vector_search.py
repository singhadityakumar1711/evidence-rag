"""Find stored chunks closest to a query embedding."""

from pgvector import Vector

from app.storage.database import get_connection


def search_chunks(
    query_embedding: list[float],
    *,
    top_k: int,
    model_name: str,
) -> list[dict]:
    """Return up to top_k chunks from the same embedding model, closest first."""
    with get_connection() as connection:
        rows = connection.execute(
            """
            SELECT chunk_id, document_id, text, metadata,
                   embedding <=> %s AS distance
            FROM chunks
            WHERE embedding_model = %s
            ORDER BY distance, chunk_id
            LIMIT %s
            """,
            (Vector(query_embedding), model_name, top_k),
        ).fetchall()

    results = []
    for chunk_id, document_id, text, metadata, distance in rows:
        results.append(
            {
                "chunk_id": chunk_id,
                "document_id": document_id,
                "text": text,
                # Convert cosine distance to similarity: higher is closer.
                "score": 1 - distance,
                "metadata": metadata,
            }
        )

    return results
