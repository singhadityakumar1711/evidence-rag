"""Load the local model and encode text as normalized vectors."""

from functools import lru_cache


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"


def load_model(model_name: str):
    """Download (if needed) and load one SentenceTransformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the ingestion model once per process."""
    return load_model(EMBEDDING_MODEL_NAME)


def embed_texts(model, texts: list[str]) -> list[list[float]]:
    """Turn a list of texts into one normalized vector per text."""
    if not texts:
        return []

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()
