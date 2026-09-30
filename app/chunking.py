"""Chunk normalized documents into retrieval-ready records."""

from dataclasses import dataclass
import re
from typing import Literal

from app.documents import Document


ChunkingStrategy = Literal["fixed_size", "sentence"]


@dataclass(frozen=True)
class Chunk:
    """A searchable slice of a document plus the metadata needed for citations."""

    chunk_id: str
    document_id: str
    text: str
    metadata: dict[str, str | int]


@dataclass(frozen=True)
class _Sentence:
    text: str
    start_char: int
    end_char: int


def chunk_document(
    document: Document,
    *,
    strategy: ChunkingStrategy,
    chunk_size: int,
    overlap: int = 0,
) -> list[Chunk]:
    """Split a normalized document using fixed-width or sentence-aware chunks.

    ``chunk_size`` and ``overlap`` are character counts. Sentence chunking keeps
    complete sentences together where possible; a sentence longer than the target
    size is emitted as its own chunk instead of being silently truncated.
    """
    _validate_parameters(chunk_size=chunk_size, overlap=overlap)

    if not document.text:
        return []

    if strategy == "fixed_size":
        spans = _fixed_size_spans(document.text, chunk_size, overlap)
    elif strategy == "sentence":
        spans = _sentence_spans(document.text, chunk_size, overlap)
    else:
        raise ValueError(f"Unsupported chunking strategy: {strategy}")

    return [
        Chunk(
            chunk_id=f"{document.document_id}:{index:04d}",
            document_id=document.document_id,
            text=document.text[start:end],
            metadata={
                "source": document.source,
                "title": document.title,
                "document_type": document.document_type,
                "chunk_index": index,
                "start_char": start,
                "end_char": end,
                "strategy": strategy,
            },
        )
        for index, (start, end) in enumerate(spans)
    ]


def _validate_parameters(*, chunk_size: int, overlap: int) -> None:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero")
    if overlap < 0:
        raise ValueError("overlap cannot be negative")
    if overlap >= chunk_size:
        raise ValueError("overlap must be smaller than chunk_size")


def _fixed_size_spans(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    step = chunk_size - overlap
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        spans.append((start, end))
        if end == len(text):
            break
        start += step
    return spans


def _sentence_spans(text: str, chunk_size: int, overlap: int) -> list[tuple[int, int]]:
    sentences = _sentences(text)
    spans: list[tuple[int, int]] = []
    current: list[_Sentence] = []

    for sentence in sentences:
        prospective_start = current[0].start_char if current else sentence.start_char
        prospective_length = sentence.end_char - prospective_start

        if current and prospective_length > chunk_size:
            spans.append((current[0].start_char, current[-1].end_char))
            current = _overlap_sentences(
                current,
                overlap=overlap,
                next_sentence=sentence,
                chunk_size=chunk_size,
            )

        current.append(sentence)

    if current:
        spans.append((current[0].start_char, current[-1].end_char))

    return spans


def _sentences(text: str) -> list[_Sentence]:
    return [
        _Sentence(match.group(), match.start(), match.end())
        for match in re.finditer(r"\S.*?[.!?](?=\s|$)|\S.*$", text)
    ]


def _overlap_sentences(
    sentences: list[_Sentence],
    *,
    overlap: int,
    next_sentence: _Sentence,
    chunk_size: int,
) -> list[_Sentence]:
    if overlap == 0:
        return []

    selected: list[_Sentence] = []
    for sentence in reversed(sentences):
        if selected and selected[0].end_char - sentence.start_char > overlap:
            break
        if next_sentence.end_char - sentence.start_char > chunk_size:
            break
        selected.insert(0, sentence)
    return selected
