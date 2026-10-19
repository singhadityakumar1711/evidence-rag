# EvidenceRAG

A learning project for ingesting text documents, storing their embeddings in PostgreSQL/pgvector, and checking whether vector search retrieves relevant evidence. It does not generate answers yet.

## What works

`POST /ingest` validates and normalizes text, splits it into **10-character fixed-size chunks**, embeds each chunk with local `all-MiniLM-L6-v2` (384 dimensions), and saves the document, chunks, source metadata, and vectors in PostgreSQL. Re-ingesting a document ID replaces its previous chunks in one transaction. The processing module also supports sentence-based chunking and configurable overlap, but the API does not currently use them.

`POST /search` embeds a question with the same model and returns the closest chunks with cosine similarity scores and source metadata. `GET /health` checks that the API is running. The API accepts document text as JSON; it does not parse uploaded PDF or HTML files. Answer generation is planned for a later step.

## Run locally

Start Docker Desktop, then run:

```powershell
docker compose up --build -d
```

Open `http://localhost:8000/docs` for the API interface, or `http://localhost:8000/health` for the health check. On first use, the embedding model downloads into a persistent Docker volume. `db/init.sql` creates the tables only when the database volume is empty; restarting Compose does not apply schema changes.

Send JSON to `POST http://localhost:8000/ingest`:

```json
{
  "document_id": "demo-001",
  "source": "manual-test",
  "title": "Local ingestion test",
  "text": "EvidenceRAG ingests documents and stores chunks for retrieval.",
  "document_type": "markdown"
}
```

The response contains `document_id` and `chunks_saved`. Stop the services with `docker compose down` (named volumes remain intact).

## Search stored documents

After ingesting documents, send JSON to `POST http://localhost:8000/search`:

```json
{
  "query": "Where are vectors stored?",
  "top_k": 3
}
```

`top_k` defaults to 5 and accepts values from 1 to 20. Blank questions and invalid limits return HTTP 422. The response contains `query` and a ranked `results` list; each result includes `chunk_id`, `document_id`, `text`, `score`, and `metadata` (source details and chunk offsets).

Search uses exact cosine distance over chunks stored with the active embedding model. Scores are `1 - distance`, ordered highest first; they are not probabilities. An empty result list means no chunks matched the model. The current 10-character chunks can produce short, weak matches; retrieval quality still needs evaluation and chunk-size tuning.

## Check the pipeline

With Compose running, execute:

```powershell
docker compose exec app uv run --no-sync python -m scripts.smoke_test_pipeline
```

The script ingests `examples/checkpoint_set.json`, checks saved counts and vector dimensions, then prints the closest chunk and source for each of three questions. It uses pgvector's cosine-distance operator (`<=>`) without an approximate index; the displayed similarity is `1 - distance`, not a probability. Results are restricted to the three sample document IDs.

An earlier run using **sentence-based** chunks saved 3 documents/9 chunks and found the expected document first for 3/3 questions. The API now uses fixed-size chunks, so those earlier counts and scores do **not** describe the current configuration; rerun the script for current results.

For the unit tests, run `uv sync` and then `uv run python -m pytest -q`. Search tests use mocked dependencies; the live `/search` request has also been manually verified by the project author. This checks functionality rather than retrieval quality. `scripts/compare_embeddings.py` is an optional local model-comparison experiment (`uv run python -m scripts.compare_embeddings`); its first run may download model weights. If running the API outside Docker, point `DATABASE_URL` at `localhost` rather than the Compose hostname `db`.

## Project layout

```text
app/main.py                 health, ingestion, and search routes
app/api/schemas.py          ingestion request/response models
app/api/search_schemas.py   search validation and result models
app/processing/             document normalization, chunking, embeddings
app/processing/retrieval.py question embedding and search orchestration
app/storage/                database connection and persistence
app/storage/vector_search.py  cosine search and result mapping
db/init.sql                 pgvector tables
examples/checkpoint_set.json
scripts/smoke_test_pipeline.py
scripts/compare_embeddings.py
tests/                      document, chunking, health, and search tests
Dockerfile, compose.yaml    local app and database services
```

Next: keyword retrieval and a common result schema, followed by hybrid search, retrieval evaluation, and answer generation with citations.
