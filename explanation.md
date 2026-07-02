# Project Explanation

This project is a FastAPI AI assistant with four integrated capabilities:

1. Knowledge ingestion (PDF/TXT/MD -> embeddings -> ChromaDB)
2. RAG-based document Q&A
3. Session memory (within current runtime)
4. Tool calling (order status + product search)

## 1 Ingestion pipeline

When a file is uploaded to `POST /ingest`, the server:

- saves it in `uploads/` as a temporary staging file
- loads content based on extension (`pypdf` for PDF, UTF-8 read for TXT/MD)
- chunks text:
  - Markdown uses section-aware splitting on `##` headers
  - other formats use fixed-size chunking with overlap
- generates embeddings with `sentence-transformers/all-MiniLM-L6-v2`
- stores vectors + original chunk text + metadata (`source`, `chunk_id`) in ChromaDB (`chroma_db/`)

## 2 Retrieval approach

Retrieval is executed only for the knowledge route (not for every message).

- Query embedding is generated with the same MiniLM model.
- Chroma cosine distance is converted to similarity (`1 - distance`).
- Results are filtered by `RETRIEVAL_SCORE_THRESHOLD`.
- Follow-up questions are expanded using recent user turns before search.
- Extra ranking heuristics are applied:
  - keyword overlap boost
  - section-header-aware fallback for short topical follow-ups (e.g., "what about loyalty program?")
- Retrieved chunks are deduplicated and returned as `sources` in chat response.

If no relevant chunk is found, the assistant returns the exact fallback sentence:
`I couldn't find that information in the uploaded documents.`

## 3 Memory implementation

Memory is implemented as an in-process dictionary keyed by `session_id`.

- Each message is stored as `{role, content}`.
- Both user and assistant turns are appended on every chat call.
- History is trimmed using a rolling window (`MEMORY_MAX_TURNS * 2` messages).
- Memory is resettable via `POST /reset/{session_id}`.
- Memory is runtime-scoped (does not persist across server restart).

## 4 Tool-calling strategy

Two mock tools are implemented:

- `get_order_status(order_id)` from `data/orders.json`
- `search_product(name)` from `data/products.json`

Routing is explicit and deterministic before generation:

- Order IDs route to order tool.
- Product/cost/catalog intent routes to product tool.
- Product routing vocabulary is derived dynamically from `products.json`, so new catalog terms do not require hardcoded updates in router logic.

Tool outputs are formatted in a dedicated tool path and intentionally separated from document context.

## 5 Prompt design

The project uses route-specific prompts:

- **Base prompt** for general chat and memory-aware responses.
- **Knowledge prompt** enforces grounded answers using only retrieved document context and instructs exact fallback when answer is absent.
- **Tool prompt** enforces isolation of tool output from document context.

Provider handling:

- Primary provider is configurable (`gemini`).
- Gemini errors are classified (`auth_error`, `rate_limit`, etc.) with retries for transient overload.
- If LLM is unavailable for knowledge route, a structured extractive fallback is used instead of dumping raw chunks.

