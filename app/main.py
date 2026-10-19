from fastapi import FastAPI, HTTPException

from app.api.schemas import IngestRequest, IngestResponse
from app.api.search_schemas import SearchRequest, SearchResponse
from app.processing.chunking import chunk_document
from app.processing.documents import ingest_text
from app.processing.embeddings import EMBEDDING_MODEL_NAME, embed_texts, get_embedding_model
from app.processing.retrieval import retrieve_chunks
from app.storage.repository import save_document


CHUNK_SIZE = 10  # Characters per fixed-size chunk, not a chunk count.

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
        strategy="fixed_size",
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


@app.post("/search", response_model=SearchResponse)
def search(request: SearchRequest) -> SearchResponse:
    """Return the closest chunks with similarity scores and source metadata."""
    results = retrieve_chunks(request.query, top_k=request.top_k)
    return SearchResponse(query=request.query, results=results) # type: ignore
