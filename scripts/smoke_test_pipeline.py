"""Smoke-test ingestion, vector storage, and first retrieval results.

Run inside the app container after starting Docker Compose:
    uv run --no-sync python -m scripts.smoke_test_pipeline
"""

import json
from pathlib import Path
from urllib.request import Request, urlopen

from pgvector import Vector

from app.processing.embeddings import embed_texts, get_embedding_model
from app.storage.database import get_connection


DATA_FILE = Path(__file__).resolve().parents[1] / "examples" / "checkpoint_set.json"
INGEST_URL = "http://127.0.0.1:8000/ingest"


def ingest_document(document: dict) -> None:
    """Send one sample document through the real ingestion API."""
    request = Request(
        INGEST_URL,
        data=json.dumps(document).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=300) as response:
        result = json.load(response)
    print(f"Ingested {result['document_id']}: {result['chunks_saved']} chunks")


def main() -> None:
    checkpoint = json.loads(DATA_FILE.read_text(encoding="utf-8"))
    documents = checkpoint["documents"]
    questions = checkpoint["questions"]
    document_ids = [document["document_id"] for document in documents]

    for document in documents:
        ingest_document(document)

    with get_connection() as connection:
        chunk_count, document_count, min_dim, max_dim = connection.execute(
            """
            SELECT COUNT(*), COUNT(DISTINCT document_id),
                   MIN(vector_dims(embedding)), MAX(vector_dims(embedding))
            FROM chunks
            WHERE document_id = ANY(%s)
            """,
            (document_ids,),
        ).fetchone()

        if document_count != len(documents) or chunk_count == 0:
            raise RuntimeError("The checkpoint documents were not all saved")
        if min_dim != 384 or max_dim != 384:
            raise RuntimeError(f"Expected 384-dimensional vectors, got {min_dim}-{max_dim}")

        print(f"\nSaved {document_count} documents and {chunk_count} chunks")
        print(f"Embedding dimensions: {min_dim}")
        print("Indexing choice: exact cosine search (no approximate vector index)")

        model = get_embedding_model()
        correct = 0
        for question in questions:
            query_vector = Vector(embed_texts(model, [question["query"]])[0])
            # Ignore unrelated documents already present in the database.
            chunk_id, document_id, text, source, distance = connection.execute(
                """
                SELECT chunk_id, document_id, text, metadata->>'source' AS source,
                       embedding <=> %s AS distance
                FROM chunks
                WHERE document_id = ANY(%s)
                ORDER BY distance
                LIMIT 1
                """,
                (query_vector, document_ids),
            ).fetchone()

            matched = document_id == question["expected_document_id"]
            correct += matched
            print(f"\n{'OK' if matched else 'MISS'} | {question['query']}")
            print(f"  Match: {document_id} / {chunk_id}")
            print(f"  Source: {source}")
            print(f"  Cosine similarity: {1 - distance:.3f}")
            print(f"  Text: {text}")

        print(f"\nExpected document at rank 1: {correct}/{len(questions)}")


if __name__ == "__main__":
    main()
