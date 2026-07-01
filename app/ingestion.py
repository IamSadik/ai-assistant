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
def chunk_text(
    text: str,
    chunk_size: int = None,
    overlap: int = None,
) -> List[str]:
    """
    Split text into overlapping fixed-size chunks.

    Whitespace is normalized first. Chunks shorter than 20 chars after
    stripping (e.g. trailing whitespace-only tail chunks) are dropped.
    """
    chunk_size = chunk_size or settings.CHUNK_SIZE
    overlap = overlap or settings.CHUNK_OVERLAP

    # Normalize whitespace so page breaks / repeated newlines from PDFs
    # don't inflate chunk boundaries with junk.
    normalized = " ".join(text.split())

    if not normalized:
        return []

    chunks = []
    start = 0
    text_len = len(normalized)

    while start < text_len:
        end = start + chunk_size
        chunk = normalized[start:end].strip()
        if len(chunk) >= 20:
            chunks.append(chunk)

        if end >= text_len:
            break
        start = end - overlap  # step forward, but re-cover the overlap window

    return chunks


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