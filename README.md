# EvidenceRAG

A learning-focused, production-style Retrieval-Augmented Generation (RAG) project for answering questions over public documents with source citations.

## Current progress

Completed so far:

- FastAPI application with a `/health` endpoint.
- Text ingestion with whitespace normalization.
- Normalized document schema: ID, source, title, document type, and text.
- Fixed-size and sentence-based chunking.
- Configurable character overlap between chunks.
- Chunk metadata for citations: source, title, document type, chunk index, and character offsets.
- Local embedding helpers using SentenceTransformers.
- A small script that compares local embedding models on ten labelled retrieval questions.

## Current architecture

```text
Document text
    ↓
Ingestion and whitespace cleaning
    ↓
Chunks + source metadata
    ↓
Local embedding model
    ↓
Cosine-similarity retrieval experiment
```

PostgreSQL + pgvector, hybrid retrieval, reranking, answer generation, and citations in generated answers are planned next.

## Local setup

Install the project dependencies:

```powershell
uv sync
```

Run the API:

```powershell
uv run uvicorn app.main:app --reload
```

Check the API:

```text
http://127.0.0.1:8000/health
```

## Run tests

```powershell
uv run pytest -q
```

## Local embedding experiment

The experiment uses a tiny public-safe document set and ten hand-labelled questions. For each model, it:

1. converts document chunks into vectors;
2. converts each question into a vector;
3. calculates cosine similarity against every chunk;
4. selects the highest-scoring chunk;
5. reports Recall@1, the fraction of questions where the correct chunk ranked first.

Configure the local models in `scripts/compare_embeddings.py`, then run:

```powershell
uv run python -m scripts.compare_embeddings
```

The first run downloads model weights into the local Hugging Face cache. Model weights are not committed to this repository.

## Project structure

```text
app/
  documents.py      # document schema and text ingestion
  chunking.py       # fixed-size and sentence chunking
  embeddings.py     # load a local model and create text vectors
  main.py           # FastAPI application
scripts/
  compare_embeddings.py
tests/
  test_documents.py
  test_chunking.py
  test_health.py
```
