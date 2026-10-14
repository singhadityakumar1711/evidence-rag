from functools import lru_cache

from fastapi import FastAPI, HTTPException

from app.chunking import chunk_document
from app.documents import ingest_text
from app.embeddings import embed_texts, load_model
from app.repository import save_document
from app.schemas import IngestRequest, IngestResponse


EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CHUNK_SIZE = 10


@lru_cache(maxsize=1)
def get_embedding_model():
    """Load the local model on the first ingest and reuse it afterward."""
    return load_model(EMBEDDING_MODEL_NAME)

app = FastAPI(
    title="EvidenceRAG",
    version="0.1.0",
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/ingest", response_model=IngestResponse)
def ingest(request: IngestRequest) -> IngestResponse:
    """Normalize, chunk, embed, and persist one text document."""
    document = ingest_text(
        document_id=request.document_id,
        source=request.source,
        title=request.title,
        text=request.text,
        document_type=request.document_type,
    )
    if not document.text:
        raise HTTPException(
            status_code=422,
            detail="Document text must not be empty after whitespace normalization",
        )

    chunks = chunk_document(
        document,
        strategy="sentence",
        chunk_size=CHUNK_SIZE,
        overlap=0,
    )
    model = get_embedding_model()
    embeddings = embed_texts(model, [chunk.text for chunk in chunks])
    chunks_saved = save_document(
        document=document,
        chunks=chunks,
        embeddings=embeddings,
        model_name=EMBEDDING_MODEL_NAME,
    )

    return IngestResponse(
        document_id=document.document_id,
        chunks_saved=chunks_saved,
    )
