import re
from dataclasses import dataclass


@dataclass
class Chunk:
    chunk_id: int
    text: str
    source: str
    length: int


def clean_text(text: str) -> str:
    """remove null bytes and collapse whitespaces"""
    text = text.replace("\x00", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
    """Split text into overlapping character chunks.
    chunk_size = Max characters per chunk
    overlap= characters shared between consecutive chunks"""
    chunks: list[str] = []
    start = 0
    total = len(text)
    while start < total:
        end = min(start + chunk_size, total)
        chunks.append(text[start:end])
        if end == total:
            break
        start += chunk_size - overlap
    return chunks


def build_chunk_records(chunks: list[str], source: str) -> list[Chunk]:
    """Attach Metadata to each chunk string"""
    return [
        Chunk(chunk_id=i, text=chunk, source=source, length=len(chunk))
        for i, chunk in enumerate(chunks)
    ]
