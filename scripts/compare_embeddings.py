"""Compare MiniLM and BGE-M3 on ten simple retrieval questions.

Run with:
    uv run python -m scripts.compare_embeddings
"""

from math import sqrt

from app.chunking import chunk_document
from app.documents import ingest_text
from app.embeddings import embed_texts, load_model


MODEL_NAMES = [
    "sentence-transformers/all-MiniLM-L6-v2",
    # "BAAI/bge-m3",
    "BAAI/bge-small-en-v1.5",
    "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
]

# Each document has one idea. The expected ID tells us which document answers
# the question. This is our tiny, hand-labelled evaluation set.
SAMPLE_DOCUMENTS = [
    ("normalization", "Whitespace is cleaned by replacing repeated spaces, tabs, and newlines with one space."),
    ("overlap", "Chunk overlap repeats the end of one chunk at the start of the next chunk to preserve context."),
    ("sentence", "Sentence chunking keeps complete sentences together whenever possible."),
    ("metadata", "Chunk metadata stores the source, title, document type, and character offsets for citations."),
    ("embedding", "An embedding converts text into a numerical vector that represents its meaning."),
    ("cosine", "Cosine similarity compares the angle between a query vector and a chunk vector."),
    ("pgvector", "PostgreSQL with pgvector will store vectors after the prototype works in memory."),
    ("hybrid", "Hybrid retrieval combines keyword search with semantic vector search."),
    ("citation", "Citations show the source document and chunk that supports an answer."),
    ("recall", "Recall at k measures whether a relevant chunk appears in the first k results."),
]

QUESTIONS = [
    ("How does ingestion handle tabs and newlines?", "normalization"),
    ("Why should neighbouring chunks share text?", "overlap"),
    ("Which chunking strategy avoids breaking a sentence?", "sentence"),
    ("What data lets us trace an answer back to its source?", "metadata"),
    ("What does an embedding model produce?", "embedding"),
    ("How do we score the closeness of two embeddings?", "cosine"),
    ("Which database extension will persist vectors?", "pgvector"),
    ("What approach mixes lexical and semantic retrieval?", "hybrid"),
    ("How can users see the evidence for an answer?", "citation"),
    ("What does Recall@k tell us?", "recall"),
]


def cosine_similarity(first_vector: list[float], second_vector: list[float]) -> float:
    """Return how similar two vectors are: 1 is similar, 0 is unrelated."""
    dot_product = sum(first * second for first, second in zip(first_vector, second_vector))
    first_length = sqrt(sum(value * value for value in first_vector))
    second_length = sqrt(sum(value * value for value in second_vector))
    return dot_product / (first_length * second_length)


def make_chunks():
    chunks = []
    for document_id, text in SAMPLE_DOCUMENTS:
        document = ingest_text(
            document_id=document_id,
            source=f"sample://{document_id}",
            title=document_id,
            text=text,
            document_type="markdown",
        )
        chunks.extend(chunk_document(document, strategy="sentence", chunk_size=300))
    return chunks


def find_best_chunk(query, chunks, chunk_embeddings, model):
    """Embed one question and return the chunk with the highest score."""
    query_embedding = embed_texts(model, [query])[0]

    best_chunk = chunks[0]
    best_score = -1.0
    for chunk, chunk_embedding in zip(chunks, chunk_embeddings):
        score = cosine_similarity(query_embedding, chunk_embedding)
        if score > best_score:
            best_chunk = chunk
            best_score = score

    return best_chunk, best_score


def compare_model(model_name: str) -> None:
    print(f"\nLoading {model_name}")
    model = load_model(model_name)
    chunks = make_chunks()
    chunk_embeddings = embed_texts(model, [chunk.text for chunk in chunks])

    correct_answers = 0
    for question, expected_document_id in QUESTIONS:
        best_chunk, score = find_best_chunk(question, chunks, chunk_embeddings, model)
        is_correct = best_chunk.document_id == expected_document_id
        correct_answers += is_correct
        print(f"{'OK' if is_correct else 'MISS'} | {question}")
        print(f"     Retrieved: {best_chunk.document_id} (score: {score:.3f})")

    accuracy = correct_answers / len(QUESTIONS)
    print(f"\nVector dimensions: {len(chunk_embeddings[0])}")
    print(f"Recall@1: {correct_answers}/{len(QUESTIONS)} = {accuracy:.0%}")


def main() -> None:
    for model_name in MODEL_NAMES:
        compare_model(model_name)


if __name__ == "__main__":
    main()
