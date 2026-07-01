# Architecture / Pipeline Diagram

This renders natively on GitHub (Mermaid support built in).

```mermaid
flowchart TD
    subgraph Ingestion["1 · Knowledge Ingestion (POST /ingest)"]
        A1[User uploads PDF / TXT / MD] --> A2[Load document\npypdf or plain text read]
        A2 --> A3[Chunk text\nfixed-size, with overlap]
        A3 --> A4[Generate embeddings\nsentence-transformers MiniLM]
        A4 --> A5[(ChromaDB\npersistent vector store)]
    end

    subgraph Chat["2-8 · Chat Pipeline (POST /chat)"]
        B1[User sends message] --> B2[Load session memory\nin-memory dict, keyed by session_id]
        B2 --> B3[Embed query &\nretrieve top-k chunks\nfrom ChromaDB]
        B3 --> B4{LLM decides:\nTool call? KB answer? Direct reply?}
        B4 -->|Order / product intent| B5[Execute tool\nget_order_status / search_product\nover orders.json / products.json]
        B4 -->|Document question| A5
        B4 -->|General / memory-based| B6[Use conversation history]
        B5 --> B7[LLM generates final response]
        A5 --> B7
        B6 --> B7
        B7 --> B8[Append reply to session memory]
        B8 --> B9[Return response to user]
    end

    A5 -.shared vector store.-> B3
```

## Text description (fallback if Mermaid doesn't render)

```
[Ingestion]
  Upload (PDF/TXT/MD)
      -> Load document (pypdf / plain text)
      -> Chunk (fixed-size, overlap)
      -> Embed (MiniLM, local)
      -> Store (ChromaDB, persisted to disk)

[Chat]
  User message
      -> Load session memory (in-memory dict)
      -> Embed query -> retrieve top-k chunks from ChromaDB
      -> LLM call with: system prompt + retrieved context + tool schemas
                        + conversation history
      -> LLM decides:
           - Tool needed (order/product)? -> call get_order_status /
             search_product -> feed result back to LLM -> final reply
           - Document question, answer in context?  -> answer from context
           - Document question, NOT in context?      -> fixed fallback message
           - General/memory-based (name recall, "cheaper options", etc.)
             -> answered from conversation history directly
      -> Append user + assistant turns to session memory
      -> Return reply (+ used_tool, used_retrieval, sources) to user
```
