"""Request and response shapes for the EvidenceRAG API."""

from pydantic import BaseModel, Field

from app.processing.documents import DocumentType


class IngestRequest(BaseModel):
    """Text and source details submitted for ingestion."""

    document_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    title: str = Field(min_length=1)
    document_type: DocumentType
    text: str = Field(min_length=1)


class IngestResponse(BaseModel):
    """Summary of a successfully stored document."""

    document_id: str
    chunks_saved: int
