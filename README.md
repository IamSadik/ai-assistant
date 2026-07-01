# Mini AI Assistant

This repository implements the AI Developer take-home assignment as a small FastAPI assistant with document ingestion, vector retrieval, session memory, and mock tool calling.

## What it does

- Upload PDF, TXT, or Markdown documents.
- Chunk and embed the text into ChromaDB.
- Chat against the uploaded knowledge base.
- Keep per-session conversation memory in-process.
- Call mock tools for order status and product lookup.
- Switch between Gemini and Ollama for answer generation.

## Project Structure

- `app/ingestion.py` loads documents, chunks text, embeds it, and stores vectors in ChromaDB.
- `app/retrieval.py` embeds the user query and fetches the most relevant chunks.
- `app/memory.py` stores session history in memory.
- `app/tools.py` reads `data/orders.json` and `data/products.json`.
- `app/llm.py` routes each request, calls tools if needed, and generates the final answer with Gemini or Ollama.
- `app/main.py` exposes the FastAPI endpoints and serves the browser UI.
- `static/index.html` contains the demo UI.

## Setup

1. Create and activate the virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure `.env`.

## Environment Variables

Use these values in `.env`:

```env
LLM_PROVIDER=gemini
GEMINI_API_KEY=your_gemini_api_key_here
GEMINI_MODEL=gemini-1.5-flash

OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama3.1

EMBEDDING_MODEL=all-MiniLM-L6-v2
CHROMA_PERSIST_DIR=./chroma_db
CHUNK_SIZE=500
CHUNK_OVERLAP=50
RETRIEVAL_TOP_K=4
RETRIEVAL_SCORE_THRESHOLD=0.3
MEMORY_MAX_TURNS=10
LLM_TEMPERATURE=0.2
```

### Provider switch

- Set `LLM_PROVIDER=gemini` to use Gemini first. This is the default recommendation for a free hosted model.
- Set `LLM_PROVIDER=ollama` to use a local Ollama model.
- If the requested provider is unavailable, the app falls back to the other configured provider and then to deterministic replies.

For Ollama, start the local server and pull a model first, for example:

```bash
ollama pull llama3.1
ollama serve
```

## Run the app

```bash
uvicorn app.main:app --reload
```

Open the browser UI at `http://127.0.0.1:8000`.

## API Endpoints

- `GET /` - browser UI
- `GET /health` - provider and knowledge-base status
- `POST /ingest` - upload a document
- `POST /chat` - send a chat message
- `GET /history/{session_id}` - inspect session memory
- `POST /reset/{session_id}` - clear a session

## How the pipeline works

1. A user uploads a document.
2. The document is loaded, chunked, embedded, and stored in ChromaDB.
3. A chat message arrives with a session ID.
4. Conversation memory is loaded for that session.
5. The request is routed to a tool, retrieval, or direct answer path.
6. Tool results and retrieved context are grounded into the final response.
7. The assistant reply is returned and appended back into memory.

## Implementation notes

- Retrieval only returns chunks above the similarity threshold, which keeps the assistant from answering from weak matches.
- Session memory is process-local, which satisfies the assignment but does not survive a server restart.
- The knowledge-base answer path returns the exact fallback sentence when no relevant content is found.
- The product tool supports the example "cheaper options" memory behavior by looking back at earlier product mentions in the session.

## Sample tools

The repository already includes `data/orders.json` and `data/products.json` with sample tool data.

## Architecture diagram

See [diagram.md](diagram.md).
