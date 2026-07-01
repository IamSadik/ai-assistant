"""
Retrieval
=========
Embeds the user's query, searches ChromaDB for the top-k nearest chunks,
and converts Chroma's cosine *distance* into a cosine *similarity* score
(1 - distance) so thresholds read naturally (higher = more relevant).

Follow-up questions ("what about the loyalty program?") are expanded with
recent conversation context before embedding, so retrieval understands what
the user is still talking about.

If the best match's similarity is below RETRIEVAL_SCORE_THRESHOLD, we treat
the knowledge base as "not having the answer" -- this is what powers the
required fallback message:
  "I couldn't find that information in the uploaded documents."
"""
import re
from typing import List, Optional, TypedDict

from app.config import settings
from app.ingestion import get_embedding_model, get_collection

_FOLLOW_UP_PREFIXES = (
    "what about",
    "how about",
    "tell me more",
    "and the",
    "anything about",
    "something about",
    "more about",
    "details on",
    "details about",
    "info on",
    "info about",
    "explain the",
    "explain",
)
_STOPWORDS = {
    "a", "an", "and", "are", "be", "but", "for", "from", "how", "i", "in",
    "is", "it", "me", "my", "of", "on", "or", "our", "tell", "the", "their",
    "this", "to", "what", "when", "where", "who", "why", "with", "you",
    "can", "could", "would", "do", "does", "did", "about", "please",
}


class RetrievedChunk(TypedDict):
    source: str
    chunk_id: int
    text: str
    score: float


def _tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _meaningful_terms(text: str) -> list[str]:
    return [t for t in _tokenize(text) if t not in _STOPWORDS and len(t) > 1]


def _is_follow_up_query(message: str) -> bool:
    text = message.lower().strip()
    if any(text.startswith(prefix) for prefix in _FOLLOW_UP_PREFIXES):
        return True
    if any(prefix in text for prefix in _FOLLOW_UP_PREFIXES):
        return True
    terms = _meaningful_terms(text)
    return len(terms) <= 4


def _topic_terms_from_message(message: str) -> list[str]:
    """Extract topic keywords from a short or follow-up message."""
    text = message.lower().strip()
    for prefix in _FOLLOW_UP_PREFIXES:
        if text.startswith(prefix):
            text = text[len(prefix) :].strip()
            break
        if prefix in text:
            idx = text.find(prefix)
            text = text[idx + len(prefix) :].strip()
            break
    return _meaningful_terms(text)


def build_retrieval_query(history: list[dict], latest_message: str) -> str:
    """
    Build an embedding query that includes conversation context for short or
    follow-up messages so retrieval stays on-topic across turns.
    """
    message = latest_message.strip()
    if not history or not _is_follow_up_query(message):
        return message

    recent_user: list[str] = []
    for turn in reversed(history):
        if turn.get("role") != "user":
            continue
        content = str(turn.get("content", "")).strip()
        if content and content != message:
            recent_user.append(content)
        if len(recent_user) >= 2:
            break

    if not recent_user:
        return message

    context = " ".join(reversed(recent_user))
    return f"{context} {message}".strip()


def _dedupe_chunks(chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    seen: set[tuple[str, int]] = set()
    unique: List[RetrievedChunk] = []
    for chunk in chunks:
        key = (chunk["source"], chunk["chunk_id"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(chunk)
    return unique


def _keyword_boost(query: str, chunks: List[RetrievedChunk]) -> List[RetrievedChunk]:
    """Slightly boost chunks whose text contains query keywords (e.g. loyalty)."""
    terms = set(_meaningful_terms(query))
    if not terms:
        return chunks

    boosted: List[RetrievedChunk] = []
    for chunk in chunks:
        text_terms = set(_tokenize(chunk["text"]))
        overlap = len(terms & text_terms)
        score = chunk["score"] + min(overlap * 0.05, 0.15)
        boosted.append(
            RetrievedChunk(
                source=chunk["source"],
                chunk_id=chunk["chunk_id"],
                text=chunk["text"],
                score=round(min(score, 1.0), 4),
            )
        )

    boosted.sort(key=lambda c: c["score"], reverse=True)
    return boosted


def _keyword_search_chunks(terms: list[str], top_k: int) -> List[RetrievedChunk]:
    """Fallback: scan stored chunks for topic keyword / section-header matches."""
    if not terms:
        return []

    collection = get_collection()
    if collection.count() == 0:
        return []

    data = collection.get(include=["documents", "metadatas"])
    documents = data.get("documents") or []
    metadatas = data.get("metadatas") or []
    term_set = set(terms)

    scored: list[tuple[float, RetrievedChunk]] = []
    for doc, meta in zip(documents, metadatas):
        if not doc:
            continue
        text_lower = doc.lower()
        header = text_lower.split("\n", 1)[0]
        body_score = sum(1.0 for t in term_set if t in text_lower)
        header_score = sum(5.0 for t in term_set if t in header)
        total = body_score + header_score
        if total <= 0:
            continue
        scored.append(
            (
                total,
                RetrievedChunk(
                    source=meta.get("source", "unknown"),
                    chunk_id=meta.get("chunk_id", -1),
                    text=doc,
                    score=round(min(0.5 + total * 0.05, 0.95), 4),
                ),
            )
        )

    scored.sort(key=lambda item: item[0], reverse=True)
    return [chunk for _, chunk in scored[:top_k]]


def retrieve_for_knowledge(
    query: str,
    history: Optional[list[dict]] = None,
    top_k: int = None,
) -> List[RetrievedChunk]:
    """
    Retrieve chunks for a knowledge-base question.

    Uses embedding search first, then keyword/section matching when the query
    is a short follow-up (e.g. "what about the loyalty program?").
    """
    top_k = top_k or settings.RETRIEVAL_TOP_K
    effective_query = (
        build_retrieval_query(history, query) if history is not None else query
    )

    chunks = retrieve(query, top_k=top_k, history=history)
    topic_terms = _topic_terms_from_message(query)
    if not topic_terms and history:
        topic_terms = _meaningful_terms(effective_query)

    if topic_terms:
        keyword_chunks = _keyword_search_chunks(topic_terms, top_k)
        if keyword_chunks:
            header_match = any(
                t in keyword_chunks[0]["text"].lower().split("\n", 1)[0]
                for t in topic_terms
            )
            if not chunks or header_match or _is_follow_up_query(query):
                return _dedupe_chunks(keyword_chunks + (chunks or []))[:top_k]

    if chunks and topic_terms:
        boosted = _keyword_boost(effective_query, chunks)
        header_hits = [
            c
            for c in boosted
            if any(t in c["text"].lower().split("\n", 1)[0] for t in topic_terms)
        ]
        if header_hits:
            return _dedupe_chunks(header_hits + boosted)[:top_k]

    if chunks:
        return chunks

    if topic_terms:
        keyword_chunks = _keyword_search_chunks(topic_terms, top_k)
        if keyword_chunks:
            return keyword_chunks

    if _is_follow_up_query(query) and history:
        relaxed = retrieve(
            effective_query,
            top_k=top_k,
            history=history,
            min_score=settings.RETRIEVAL_SCORE_THRESHOLD * 0.65,
        )
        if relaxed:
            return relaxed

    return []


def retrieve(
    query: str,
    top_k: int = None,
    history: Optional[list[dict]] = None,
    min_score: Optional[float] = None,
) -> List[RetrievedChunk]:
    """
    Retrieve the top-k most relevant chunks for a query, filtered by the
    minimum similarity threshold. Returns an empty list if the knowledge
    base is empty or nothing clears the threshold.
    """
    top_k = top_k or settings.RETRIEVAL_TOP_K
    threshold = min_score if min_score is not None else settings.RETRIEVAL_SCORE_THRESHOLD
    effective_query = query
    if history is not None:
        effective_query = build_retrieval_query(history, query)

    collection = get_collection()
    if collection.count() == 0:
        return []

    model = get_embedding_model()
    query_embedding = model.encode([effective_query]).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=min(top_k * 2, collection.count()),
    )

    documents = results.get("documents", [[]])[0]
    metadatas = results.get("metadatas", [[]])[0]
    distances = results.get("distances", [[]])[0]

    retrieved: List[RetrievedChunk] = []
    for doc, meta, dist in zip(documents, metadatas, distances):
        similarity = 1 - dist
        if similarity >= threshold:
            retrieved.append(
                RetrievedChunk(
                    source=meta.get("source", "unknown"),
                    chunk_id=meta.get("chunk_id", -1),
                    text=doc,
                    score=round(similarity, 4),
                )
            )

    retrieved = _dedupe_chunks(retrieved)
    retrieved = _keyword_boost(effective_query, retrieved)
    return retrieved[:top_k]


def format_context(chunks: List[RetrievedChunk]) -> str:
    """Format retrieved chunks into a context block for the LLM prompt."""
    if not chunks:
        return "(no relevant context found in the knowledge base)"

    parts = []
    for c in chunks:
        parts.append(f"[Source: {c['source']}, chunk {c['chunk_id']}]\n{c['text']}")
    return "\n\n---\n\n".join(parts)
