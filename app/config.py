"""
Centralized configuration for the AI Assistant.
"""
import os
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=True)


def _env(name: str, default: str = "") -> str:
    """Read an env var, strip whitespace and optional surrounding quotes."""
    raw = os.getenv(name, default)
    if raw is None:
        return default
    value = raw.strip()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in ("'", '"'):
        value = value[1:-1].strip()
    return value


def validate_gemini_key(key: str) -> Optional[str]:
    """Return a human-readable hint when the API key format looks wrong."""
    if not key:
        return "GEMINI_API_KEY is not set in .env"
    if not key.startswith("AIza"):
        return (
            "GEMINI_API_KEY format looks invalid (expected Google AI Studio key "
            "starting with AIza). Create one at https://aistudio.google.com/apikey"
        )
    return None


class Settings:
    # --- LLM ---
    @property
    def LLM_PROVIDER(self) -> str:
        return _env("LLM_PROVIDER", "gemini").lower()

    @property
    def GEMINI_API_KEY(self) -> str:
        return _env("GEMINI_API_KEY")

    @property
    def GEMINI_MODEL(self) -> str:
        return _env("GEMINI_MODEL", "gemini-2.5-flash")

    @property
    def OLLAMA_BASE_URL(self) -> str:
        return _env("OLLAMA_BASE_URL", "http://localhost:11434")

    @property
    def OLLAMA_MODEL(self) -> str:
        return _env("OLLAMA_MODEL", "llama3.1")

    @property
    def LLM_TEMPERATURE(self) -> float:
        return float(_env("LLM_TEMPERATURE", "0.2"))

    # --- Embeddings ---
    @property
    def EMBEDDING_MODEL(self) -> str:
        return _env("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # --- Vector store ---
    CHROMA_PERSIST_DIR: str = _env("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = "knowledge_base"

    # --- Chunking ---
    @property
    def CHUNK_SIZE(self) -> int:
        return int(_env("CHUNK_SIZE", "500"))

    @property
    def CHUNK_OVERLAP(self) -> int:
        return int(_env("CHUNK_OVERLAP", "50"))

    # --- Retrieval ---
    @property
    def RETRIEVAL_TOP_K(self) -> int:
        return int(_env("RETRIEVAL_TOP_K", "4"))

    @property
    def RETRIEVAL_SCORE_THRESHOLD(self) -> float:
        return float(_env("RETRIEVAL_SCORE_THRESHOLD", "0.3"))

    # --- Memory ---
    @property
    def MEMORY_MAX_TURNS(self) -> int:
        return int(_env("MEMORY_MAX_TURNS", "10"))

    # --- Paths ---
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    ORDERS_FILE: str = os.path.join(DATA_DIR, "orders.json")
    PRODUCTS_FILE: str = os.path.join(DATA_DIR, "products.json")
    UPLOADS_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "uploads"
    )


settings = Settings()

os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)
