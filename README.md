# EvidenceRAG

A learning project for ingesting text documents, storing their embeddings in PostgreSQL/pgvector, and checking whether vector search retrieves relevant evidence. It does not generate answers yet.

## What works

`POST /ingest` validates and normalizes text, splits it into **10-character fixed-size chunks**, embeds each chunk with local `all-MiniLM-L6-v2` (384 dimensions), and saves the document, chunks, source metadata, and vectors in PostgreSQL. Re-ingesting a document ID replaces its previous chunks in one transaction. The processing module also supports sentence-based chunking and configurable overlap, but the API does not currently use them.

`GET /health` checks that the API is running. The API accepts document text as JSON; it does not parse uploaded PDF or HTML files. A standalone smoke test embeds questions and performs exact cosine search over three sample documents. There is no public search or answer-generation endpoint yet.

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

## Check the pipeline

With Compose running, execute:

```powershell
docker compose exec app uv run --no-sync python -m scripts.smoke_test_pipeline
```

The script ingests `examples/checkpoint_set.json`, checks saved counts and vector dimensions, then prints the closest chunk and source for each of three questions. It uses pgvector's cosine-distance operator (`<=>`) without an approximate index; the displayed similarity is `1 - distance`, not a probability. Results are restricted to the three sample document IDs.

An earlier run using **sentence-based** chunks saved 3 documents/9 chunks and found the expected document first for 3/3 questions. The API now uses fixed-size chunks, so those earlier counts and scores do **not** describe the current configuration; rerun the script for current results.

For the unit tests, run `uv sync` and then `uv run pytest -q`. `scripts/compare_embeddings.py` is an optional local model-comparison experiment (`uv run python -m scripts.compare_embeddings`); its first run may download model weights. If running the API outside Docker, point `DATABASE_URL` at `localhost` rather than the Compose hostname `db`.

## Project layout

```text
app/main.py                 API routes and ingestion wiring
app/api/schemas.py          request/response models
app/processing/             document normalization, chunking, embeddings
app/storage/                database connection and persistence
db/init.sql                 pgvector tables
examples/checkpoint_set.json
scripts/smoke_test_pipeline.py
scripts/compare_embeddings.py
tests/                      document, chunking, and health tests
Dockerfile, compose.yaml    local app and database services
```

Next: reusable top-k search, retrieval evaluation, and answer generation with citations.
