# Architecture / Pipeline Diagram

Architecture for the AI-Assistant.

```mermaid
flowchart TD
    subgraph Ingestion["1 · Knowledge Ingestion (POST /ingest)"]
        A1[User uploads PDF / TXT / MD] --> A2[Save to uploads/ staging file]
        A2 --> A3[Load document\npypdf or UTF-8 read]
        A3 --> A4[Chunk text\nsection-aware for Markdown,\nfixed-size with overlap otherwise]
        A4 --> A5[Generate embeddings\nsentence-transformers all-MiniLM-L6-v2]
        A5 --> A6[(ChromaDB\nchroma_db/ persistent store)]
    end

    subgraph Chat["2 · Chat Pipeline (POST /chat)"]
        B1[User sends message + session_id] --> B2[Load session memory\nin-memory dict, rolling window]
        B2 --> B3{Route message\nno embedding yet}

        B3 -->|Order ID e.g. ORD001| T1[get_order_status\norders.json]
        B3 -->|Product / catalog intent| T2[search_product\nproducts.json]
        B3 -->|Name recall / greeting| M1[Answer from session memory\nno retrieval]
        B3 -->|Document / policy question| R1[Retrieve from ChromaDB]
        B3 -->|General chat| D1[Direct LLM reply\nhistory only]

        R1 --> R2[Build query\nexpand follow-ups with recent context]
        R2 --> R3[Embed query with MiniLM]
        R3 --> R4[Vector search top-k chunks\n+ keyword section fallback]
        R4 --> R5{Chunks found?}
        R5 -->|Yes| L1[LLM answer grounded in retrieved context]
        R5 -->|No| L2[Fixed fallback:\nI couldn't find that information...]

        T1 --> F1[Format tool result]
        T2 --> F1
        M1 --> F2[Final reply]
        L1 --> F2
        L2 --> F2
        D1 --> F2
        F1 --> F2

        F2 --> B4[Append user + assistant turns\nto session memory]
        B4 --> B5[Return reply\n+ used_tool / used_retrieval / sources]
    end

    A6 -.queried only on knowledge route.-> R3

    style R1 fill:#e8f4ea
    style R2 fill:#e8f4ea
    style R3 fill:#e8f4ea
    style R4 fill:#e8f4ea
    style T1 fill:#e8eef8
    style T2 fill:#e8eef8
    style M1 fill:#f5f0e8
```

## Key design points

| Path | Embedding / ChromaDB? | Data source |
|------|----------------------|-------------|
| Document questions (RAG) | **Yes** — query embedded, vector search in ChromaDB | Uploaded documents |
| Order status | **No** | `data/orders.json` |
| Product search | **No** | `data/products.json` (independent of uploaded docs) |
| Name recall / greeting | **No** | Session memory only |
| Direct chat | **No** | LLM + session memory |

**`uploads/`** — temporary copy of the file at ingest time; not searched at chat time.

**`chroma_db/`** — the vector database searched during the knowledge route only.

## Text description (fallback if Mermaid doesn't render)

```
[Ingestion — POST /ingest]
  User uploads PDF / TXT / MD
      -> Save to uploads/ (staging)
      -> Load document (pypdf / plain text)
      -> Chunk (Markdown: split on ## sections; else fixed-size with overlap)
      -> Embed chunks (all-MiniLM-L6-v2, local)
      -> Store vectors + text + metadata in ChromaDB (chroma_db/)

[Chat — POST /chat]
  User message + session_id
      -> Load session memory (in-memory dict, capped turns)
      -> ROUTE the message (no embedding for most paths):
           |
           |-- Order ID detected?
           |       -> get_order_status(orders.json) -> format reply
           |
           |-- Product / catalog intent?
           |       -> search_product(products.json) -> format reply
           |       (catalog is separate from uploaded documents)
           |
           |-- Name recall / greeting?
           |       -> answer from session memory only
           |
           |-- Document / policy / follow-up question?
           |       -> Build retrieval query (expand short follow-ups with context)
           |       -> Embed query -> vector search ChromaDB (top-k, threshold)
           |       -> Keyword/section fallback if needed
           |       -> LLM generates answer from retrieved chunks
           |       -> If nothing relevant: fixed "I couldn't find..." message
           |
           '-- Otherwise
                   -> Direct LLM reply using conversation history
      -> Append user + assistant turns to session memory
      -> Return reply (+ used_tool, used_retrieval, sources)
```

## Component map

| Module | Responsibility |
|--------|----------------|
| `app/main.py` | FastAPI endpoints, orchestrates ingest + chat |
| `app/ingestion.py` | Load → chunk → embed → ChromaDB |
| `app/retrieval.py` | Embed query, vector search, keyword fallback (knowledge route only) |
| `app/llm.py` | Route message, call tools / retrieval / memory / LLM |
| `app/memory.py` | Per-session conversation history |
| `app/tools.py` | Mock order + product lookups from JSON files |
