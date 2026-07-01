"""
Pydantic models for request/response validation across the API.
"""
from typing import List, Optional
from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., description="Unique ID for the chat session")
    message: str = Field(..., description="The user's message")


class SourceChunk(BaseModel):
    source: str
    chunk_id: int
    text: str
    score: float


class ChatResponse(BaseModel):
    session_id: str
    reply: str
    used_tool: Optional[str] = None
    used_retrieval: bool = False
    sources: List[SourceChunk] = []
    llm_degraded: bool = False
    llm_error: Optional[str] = None


class IngestResponse(BaseModel):
    filename: str
    chunks_created: int
    message: str


class HistoryTurn(BaseModel):
    role: str
    content: str


class HistoryResponse(BaseModel):
    session_id: str
    history: List[HistoryTurn]