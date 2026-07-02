# AI Assistant

A FastAPI-based assistant that combines document Q&A (RAG), session memory, and mock tool calling for order lookup and product search. Upload PDF, TXT, or Markdown files, chat against the knowledge base, and query a separate product catalog or order system through the same interface.

## Features

- **Knowledge ingestion** — chunk, embed, and store documents in ChromaDB
- **RAG chat** — answer questions from uploaded files with source citations
- **Session memory** — remember context within a chat session (names, follow-up questions)
- **Tool calling** — order status and product catalog lookups from JSON files
- **Web UI** — simple browser interface at `/`

## Project structure

| Path | Role |
|------|------|
| `app/ingestion.py` | Load, chunk, embed, and store documents |
| `app/retrieval.py` | Vector search against ChromaDB (knowledge route only) |
| `app/llm.py` | Message routing, tool execution, answer generation |
| `app/memory.py` | Per-session conversation history |
| `app/tools.py` | Order and product catalog tools |
| `app/main.py` | FastAPI app and endpoints |
| `static/index.html` | Browser UI |
| `data/` | Sample `orders.json` and `products.json` |
| `diagram.md` | Architecture diagram |

## Setup

### 1. Clone the repository

```bash
git clone https://github.com/IamSadik/ai-assistant.git
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

**Windows (PowerShell)**

```bash
.\venv\Scripts\Activate.ps1
```

**macOS / Linux**

```bash
source venv/bin/activate
```

### 3. Configure environment variables

Copy the `.env` file provided in your email into the project root.

You can also start from the included template:

```bash
cp .env.example .env
```

Then edit `.env` and set your API key and any other values. The template mirrors the current project defaults:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-2.5-flash

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=4
RETRIEVAL_SCORE_THRESHOLD=0.4
MEMORY_MAX_TURNS=20
LLM_TEMPERATURE=0.2
```

Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/apikey) if you need one.

### 4. Install dependencies

```bash
pip install -r requirements.txt
```

### 5. Run the server

```bash
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.

## LLM provider

The app uses **Gemini** by default (`LLM_PROVIDER=gemini`). Set `LLM_PROVIDER=ollama` to use a local [Ollama](https://ollama.com/) model instead. If the primary provider is unavailable, the app falls back to structured deterministic replies where possible.

## API endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/` | Chat UI |
| `GET` | `/health` | Server, LLM, and knowledge-base status |
| `POST` | `/ingest` | Upload a PDF, TXT, or MD document |
| `POST` | `/chat` | Send a message (`session_id`, `message`) |
| `GET` | `/history/{session_id}` | View session history |
| `POST` | `/reset/{session_id}` | Clear a session |

Interactive API docs: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

## How the pipeline works

1. **Upload** — a document is saved to `uploads/`, chunked, embedded with `all-MiniLM-L6-v2`, and stored in ChromaDB (`chroma_db/`).
2. **Chat** — each message loads session memory, then is **routed** before any embedding:
   - **Order ID** (e.g. `ORD001`) → `orders.json` lookup
   - **Catalog intent** → `products.json` search (independent of uploaded documents)
   - **Document / policy question** → vector search in ChromaDB, then LLM answer
   - **Name recall / greeting** → session memory only
   - **Other** → direct LLM reply
3. **Respond** — the reply is saved to session memory and returned with metadata (`used_tool`, `used_retrieval`, `sources`).

See [diagram.md](diagram.md) for the full architecture diagram.

## Implementation notes

- **Routing before retrieval** — only knowledge-base questions trigger embedding and ChromaDB search. Tool calls and memory-based replies do not.
- **Catalog detection** — product routing reads searchable terms from `products.json` at runtime, so new products do not require code changes.
- **Catalog vs RAG** — product results come from `products.json` only; they are not mixed with uploaded document content.
- **Retrieval threshold** — chunks below `RETRIEVAL_SCORE_THRESHOLD` are filtered out to reduce weak matches.
- **Follow-up questions** — short messages are expanded with recent conversation context before search.
- **Session memory** — stored in-process; it does not survive a server restart.

## Sample tools

### Order status (`data/orders.json`)

| Order ID | Status | Est. delivery |
|----------|--------|---------------|
| ORD001 | Shipped | 2026-07-02 |
| ORD002 | Processing | 2026-07-05 |
| ORD003 | Delivered | 2026-06-28 |
| ORD004 | Cancelled | — |

Example: *“Where is my order ORD001?”*

### Product search (`data/products.json`)

The catalog includes mice, keyboards, laptops, monitors, hubs, webcams, speakers, and more. Prices and stock are read live from the JSON file.

Examples:

- *“Do you have a wireless mouse?”*
- *“Show me some laptops and keyboards”*
- *“Show some cheaper options”* (uses prior product context in the session)

## Sample usage
- See the [demo.png](data/demo.png) file.

## Architecture diagram

See [diagram.md](diagram.md).


## Explanations
See [explanation.md](explanation.md) for brief explanation of: ingestion pipeline, retrieval approach, memory implementation, tool-calling strategy, and prompt design