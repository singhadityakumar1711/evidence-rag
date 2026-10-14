"""Save normalized documents and their embedded chunks in Postgres."""

from pgvector import Vector
from psycopg.types.json import Jsonb

from app.chunking import Chunk
from app.database import get_connection
from app.documents import Document


def save_document(
    document: Document,
    chunks: list[Chunk],
    embeddings: list[list[float]],
    model_name: str,
) -> int:
    """Insert or replace a document and all of its chunks atomically."""
    if len(chunks) != len(embeddings):
        raise ValueError("There must be one embedding for each chunk")

    with get_connection() as connection:
        connection.execute(
            """
            INSERT INTO documents
                (document_id, source, title, document_type, text)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (document_id) DO UPDATE SET
                source = EXCLUDED.source,
                title = EXCLUDED.title,
                document_type = EXCLUDED.document_type,
                text = EXCLUDED.text
            """,
            (
                document.document_id,
                document.source,
                document.title,
                document.document_type,
                document.text,
            ),
        )

        # Remove the prior version's chunks. The transaction means the old
        # rows return if saving any part of the new version fails.
        connection.execute(
            "DELETE FROM chunks WHERE document_id = %s",
            (document.document_id,),
        )

        for chunk, embedding in zip(chunks, embeddings, strict=True):
            connection.execute(
                """
                INSERT INTO chunks
                    (chunk_id, document_id, chunk_index, text,
                     embedding, embedding_model, metadata)
                VALUES (%s, %s, %s, %s, %s, %s, %s)
                """,
                (
                    chunk.chunk_id,
                    chunk.document_id,
                    chunk.metadata["chunk_index"],
                    chunk.text,
                    Vector(embedding),
                    model_name,
                    Jsonb(chunk.metadata),
                ),
            )

    return len(chunks)
