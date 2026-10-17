"""Normalize text and keep document source details."""

from dataclasses import dataclass
from typing import Literal


DocumentType = Literal["pdf", "html", "markdown"]


@dataclass
class Document:
    document_id: str
    source: str
    document_type: DocumentType
    title: str
    text: str


def clean_text(text: str) -> str:
    """Collapse whitespace so chunk offsets refer to the saved text."""
    return " ".join(text.split())


def ingest_text(
    *,
    document_id: str,
    source: str,
    title: str,
    text: str,
    document_type: DocumentType,
) -> Document:
    cleaned_text = clean_text(text)

    return Document(
        document_id=document_id,
        source=source,
        document_type=document_type,
        title=title,
        text=cleaned_text,
    )
