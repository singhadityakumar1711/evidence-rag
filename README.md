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
- Local embeddings using `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
- PostgreSQL + pgvector persistence for documents, chunks, embeddings, and metadata.
- A `POST /ingest` endpoint that chunks and embeds a document, then saves it to the database.
- Docker Compose services for the API and PostgreSQL, with persistent database and model-cache volumes.
- A reproducible checkpoint with three sample documents and three questions to verify ingestion and first vector-search results.
- A small script that compares local embedding models on ten labelled retrieval questions.

## Current architecture

```text
POST /ingest
    ↓
Validate and normalize document text
    ↓
Sentence-based chunks + source metadata
    ↓
Local all-MiniLM-L6-v2 embeddings
    ↓
PostgreSQL + pgvector

Checkpoint question → same embedding model → exact cosine search → closest chunk + source

Separate experiment: compare embedding models with labelled retrieval questions
```

Reusable top-k retrieval for user questions, hybrid search, reranking, answer generation, and citations in generated answers are planned next.

## Run the API and database with Docker

Start Docker Desktop, then run from the project directory:

```powershell
docker compose up --build
```

Compose starts the FastAPI app and PostgreSQL with pgvector. On the first database initialization, `db/init.sql` creates the extension and tables. The database files and downloaded Hugging Face model are kept in named Docker volumes, so they persist across container restarts. The model downloads on the first successful ingestion request.

Check the API health:

```text
http://localhost:8000/health
```

Interactive API docs are available at:

```text
http://localhost:8000/docs
```

### Ingest a document

Send a `POST` request to `http://localhost:8000/ingest` with a JSON body. For example:

```json
{
  "document_id": "demo-001",
  "source": "manual-test",
  "title": "Local ingestion test",
  "text": "EvidenceRAG ingests documents and divides their text into chunks. Each chunk is converted into an embedding for later retrieval.",
  "document_type": "markdown"
}
```

The response contains the document ID and number of chunks saved. Ingesting again with the same `document_id` replaces that document's previous chunks.

> `init.sql` runs only when PostgreSQL initializes an empty data volume. Schema changes to an existing database need a migration; restarting Compose does not rerun the initialization script.

## October 4 checkpoint

The original sample set in `examples/checkpoint_set.json` contains three documents and three questions with expected document IDs. With Docker Compose running, run this in another terminal:

```powershell
docker compose exec app uv run --no-sync python -m scripts.checkpoint_oct4
```

The script sends each document through `POST /ingest`, then checks that all three documents have saved chunks with 384-dimensional embeddings. For each question, it embeds the query and prints the closest chunk, its source, cosine similarity, and whether the chunk belongs to the expected document. It searches only the checkpoint document IDs, so other documents already in your database do not affect these results. Re-running the script replaces the same three documents' chunks rather than adding duplicates.

**Indexing choice:** `db/init.sql` defines `embedding VECTOR(384)` but does not create an approximate vector index. The checkpoint uses exact nearest-neighbor search with pgvector's cosine-distance operator (`<=>`), which is simple and appropriate for this tiny set. The displayed cosine similarity is `1 - distance`; it is a ranking score, not a probability. An approximate index can be evaluated later with a larger document collection.

In the recorded checkpoint run, ingestion saved **3 documents and 9 chunks**. All saved embeddings had **384 dimensions**. The first result for each question was:

| Question | Closest chunk | Source | Cosine similarity |
|---|---|---|---:|
| How does the system handle tabs and newlines in uploaded text? | `checkpoint-ingestion:0000` | `sample://checkpoint/ingestion` | 0.560 |
| Why avoid cutting a sentence in two? | `checkpoint-chunking:0000` | `sample://checkpoint/chunking` | 0.587 |
| Where are the numerical representations of chunks stored? | `checkpoint-vectors:0000` | `sample://checkpoint/vectors` | 0.535 |

The expected document appeared first for **3/3 questions** in this small sample. This is a document-level check, not a measure of answer quality: the third result points to the correct document, but its top chunk describes the vector rather than where it is stored.

## Local development and tests

Install the Python dependencies:

```powershell
uv sync
```

The API's database configuration must point to a reachable PostgreSQL instance. When running the API outside Docker while the Compose database is running, use `localhost` as the database host instead of the Compose-only hostname `db`.

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
  database.py       # PostgreSQL connection setup
  repository.py     # persist documents, chunks, embeddings, and metadata
  schemas.py        # API request and response models
  main.py           # FastAPI application and /health and /ingest endpoints
db/
  init.sql          # pgvector extension and database tables
examples/
  checkpoint_set.json  # three documents and labelled checkpoint questions
Dockerfile          # API container image
compose.yaml        # API and PostgreSQL services
scripts/
  checkpoint_oct4.py  # ingestion, dimension check, and first retrieval results
  compare_embeddings.py
tests/
  test_documents.py
  test_chunking.py
  test_health.py
```
