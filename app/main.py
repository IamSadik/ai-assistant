"""
FastAPI application entry point.

Endpoints:
  GET  /                -> minimal chat UI (static HTML)
  GET  /health           -> health check + KB stats
  POST /ingest            -> upload a PDF/TXT/MD file for ingestion
  POST /chat               -> send a chat message, get a response
  GET  /history/{session_id} -> inspect a session's memory (debugging)
  POST /reset/{session_id}   -> clear a session's memory
"""
import os
import shutil
import uuid

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from app.config import settings
from app.schemas import (
    ChatRequest,
    ChatResponse,
    IngestResponse,
    HistoryResponse,
    HistoryTurn,
    SourceChunk,
)
from app import memory, ingestion, llm

app = FastAPI(
    title="Mini AI Assistant",
    description=(
        "Knowledge ingestion + RAG chat + session memory + tool calling, "
        "built for the AI Developer take-home assignment."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", response_class=HTMLResponse)
def root():
    index_path = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_path):
        with open(index_path, "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse("<h1>AI Assistant API</h1><p>See /docs for the API.</p>")


@app.get("/health")
def health():
    return {
        "status": "ok",
        "llm_mode": "openai" if settings.OPENAI_API_KEY else "rule_based_fallback",
        "knowledge_base": ingestion.collection_stats(),
        "active_sessions": len(memory.list_sessions()),
    }


# ---------------------------------------------------------------------------
# Knowledge Ingestion
# ---------------------------------------------------------------------------
@app.post("/ingest", response_model=IngestResponse)
async def ingest(file: UploadFile = File(...)):
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in (".pdf", ".txt", ".md"):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{ext}'. Only .pdf, .txt, .md are allowed.",
        )

    # Save upload to a temp path, then run the ingestion pipeline on it.
    temp_name = f"{uuid.uuid4().hex}{ext}"
    temp_path = os.path.join(settings.UPLOADS_DIR, temp_name)

    try:
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        chunks_created = ingestion.ingest_file(temp_path, source_name=file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    finally:
        file.file.close()

    if chunks_created == 0:
        return IngestResponse(
            filename=file.filename,
            chunks_created=0,
            message="No extractable text found in the file (it may be empty or a scanned/image-only PDF).",
        )

    return IngestResponse(
        filename=file.filename,
        chunks_created=chunks_created,
        message=f"Successfully ingested '{file.filename}' into the knowledge base.",
    )


# ---------------------------------------------------------------------------
# Chat
# ---------------------------------------------------------------------------
@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    if not request.message.strip():
        raise HTTPException(status_code=400, detail="Message cannot be empty.")

    # Step 4: load conversation memory
    history = memory.get_history(request.session_id)

    # Add the new user turn to history BEFORE calling the LLM so it has
    # full context (including the current message) in one place.
    memory.append_message(request.session_id, "user", request.message)

    # Steps 5-7: decide (retrieval/tool/direct) + execute + generate
    result = llm.generate_response(
        history=memory.get_history(request.session_id),
        latest_message=request.message,
    )

    # Persist the assistant's reply into memory
    memory.append_message(request.session_id, "assistant", result["reply"])

    return ChatResponse(
        session_id=request.session_id,
        reply=result["reply"],
        used_tool=result.get("used_tool"),
        used_retrieval=result.get("used_retrieval", False),
        sources=[SourceChunk(**s) for s in result.get("sources", [])],
    )


# ---------------------------------------------------------------------------
# Session utilities (handy for debugging / demoing memory)
# ---------------------------------------------------------------------------
@app.get("/history/{session_id}", response_model=HistoryResponse)
def get_history(session_id: str):
    history = memory.get_history(session_id)
    return HistoryResponse(
        session_id=session_id,
        history=[HistoryTurn(**m) for m in history],
    )


@app.post("/reset/{session_id}")
def reset_session(session_id: str):
    memory.clear_session(session_id)
    return {"message": f"Session '{session_id}' cleared."}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)