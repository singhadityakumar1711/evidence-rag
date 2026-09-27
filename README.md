# EvidenceRAG

Production-style RAG system for answering questions over public documents
with grounded citations.

## Current Status

Day 1 — FastAPI skeleton 

## Planned Architecture

Documents
→ Parsing
→ Chunking
→ Embeddings
→ PostgreSQL + pgvector
→ Hybrid Retrieval
→ Reranking
→ Context Building
→ LLM
→ Answer + Citations

## Local Development

```bash
uv run uvicorn app.main:app --reload