"""Small helpers for creating local text embeddings."""


def load_model(model_name: str):
    """Download (if needed) and load one SentenceTransformers model."""
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


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
