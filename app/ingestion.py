"""
Knowledge Ingestion Pipeline
============================
Load -> Chunk -> Embed -> Store, for PDF / TXT / MD files.

Design choices (documented for the "brief explanation" deliverable):

- Loading: branches on file extension. PDFs use pypdf to extract text
  page by page; TXT/MD are read directly as UTF-8.
- Chunking: fixed-size character chunking with overlap (not semantic/
  sentence-aware chunking). This is a deliberate simplicity/robustness
  tradeoff -- it works uniformly across all three file types without
  needing a sentence tokenizer, and overlap (default 50 chars) prevents
  answers from being cut off exactly at a chunk boundary. A stretch goal
  would be recursive splitting on paragraph/sentence boundaries first
  (LangChain's RecursiveCharacterTextSplitter does this) before falling
  back to hard character splits.
- Embeddings: sentence-transformers/all-MiniLM-L6-v2, run locally. Chosen
  over an OpenAI embedding API so the ingestion pipeline works with zero
  API cost and zero network dependency on OpenAI specifically.
- Storage: ChromaDB, persisted to disk (CHROMA_PERSIST_DIR). Chosen over
  FAISS because Chroma bundles metadata storage + persistence + a query
  API out of the box, which keeps this file short; FAISS would need a
  separate metadata store (e.g. a pickle/SQLite side-table) to track which
  chunk/source each vector belongs to.
"""
from __future__ import annotations

import os
import uuid
from typing import List, Optional

import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer

from app.config import settings

# ---------------------------------------------------------------------------
# Lazy singletons (loaded once, reused across requests)
# ---------------------------------------------------------------------------
_embedding_model: Optional[SentenceTransformer] = None
_chroma_client = None
_collection = None


def get_embedding_model() -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        _embedding_model = SentenceTransformer(settings.EMBEDDING_MODEL)
    return _embedding_model


def get_collection():
    global _chroma_client, _collection
    if _collection is None:
        _chroma_client = chromadb.PersistentClient(path=settings.CHROMA_PERSIST_DIR)
        _collection = _chroma_client.get_or_create_collection(
            name=settings.CHROMA_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


# ---------------------------------------------------------------------------
# Step 1: Load
# ---------------------------------------------------------------------------
def load_document(file_path: str) -> str:
    """Extract raw text from a PDF, TXT, or MD file."""
    ext = os.path.splitext(file_path)[1].lower()

    if ext == ".pdf":
        reader = PdfReader(file_path)
        pages_text = []
        for page in reader.pages:
            text = page.extract_text() or ""
            pages_text.append(text)
        return "\n".join(pages_text)

    elif ext in (".txt", ".md"):
        with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
            return f.read()

    else:
        raise ValueError(
            f"Unsupported file type '{ext}'. Only .pdf, .txt, .md are supported."
        )


# ---------------------------------------------------------------------------
# Step 2: Chunk
# ---------------------------------------------------------------------------
def _split_markdown_sections(text: str) -> List[str]:
    """Split markdown on ## headings so each chunk stays topic-coherent."""
    import re

    stripped = text.strip()
    stripped = re.sub(r"^#\s+[^\n]+\n*", "", stripped, count=1).strip()

    parts = re.split(r"(?m)^##\s+", stripped)
    sections: List[str] = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        if i == 0 and not part.startswith("#"):
            if len(part.strip()) >= 20:
                sections.append(part)
            continue
        sections.append(f"## {part}")
    return sections


def _chunk_fixed(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Fixed-size overlapping chunks on normalized text."""
    normalized = " ".join(text.split())
    if not normalized:
        return []

    chunks: List[str] = []
    start = 0
    text_len = len(normalized)

    while start < text_len:
        end = start + chunk_size
        chunk = normalized[start:end].strip()
        if len(chunk) >= 20:
            chunks.append(chunk)
        if end >= text_len:
            break
        start = end - overlap

    return chunks


def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> List[str]:
    """
    Split text into chunks for embedding.

    For markdown, split on ## section headings first so each chunk covers
    one policy/topic (Return Policy, Loyalty Program, etc.). Sections
    longer than chunk_size are further split with overlap. Other formats
    use fixed-size character chunking only.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    if not text or not text.strip():
        return []

    if "## " in text:
        chunks: List[str] = []
        for section in _split_markdown_sections(text):
            if len(section) <= chunk_size:
                if len(section.strip()) >= 20:
                    chunks.append(section.strip())
            else:
                chunks.extend(_chunk_fixed(section, chunk_size, overlap))
        if chunks:
            return chunks

    return _chunk_fixed(text, chunk_size, overlap)


# ---------------------------------------------------------------------------
# Step 3 + 4: Embed + Store
# ---------------------------------------------------------------------------
def ingest_file(file_path: str, source_name: str) -> int:
    """
    Full ingestion pipeline for one file: load -> chunk -> embed -> store.
    Returns the number of chunks created and stored.
    """
    raw_text = load_document(file_path)
    chunks = chunk_text(raw_text)

    if not chunks:
        return 0

    model = get_embedding_model()
    embeddings = model.encode(chunks, show_progress_bar=False).tolist()

    collection = get_collection()

    ids = [f"{source_name}-{uuid.uuid4().hex[:8]}-{i}" for i in range(len(chunks))]
    metadatas = [
        {"source": source_name, "chunk_id": i} for i in range(len(chunks))
    ]

    collection.add(
        ids=ids,
        embeddings=embeddings,
        documents=chunks,
        metadatas=metadatas,
    )

    return len(chunks)


def collection_stats() -> dict:
    """Return basic stats about the current knowledge base (for /health or debugging)."""
    collection = get_collection()
    return {"total_chunks": collection.count()}