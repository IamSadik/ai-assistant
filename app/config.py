"""
Centralized configuration for the AI Assistant.

"""
import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    # --- LLM ---
    OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "").strip()
    OPENAI_MODEL: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")

    GEMINI_API_KEY: str = os.getenv("GEMINI_API_KEY", "").strip()
    GEMINI_MODEL: str = os.getenv("GEMINI_MODEL", "gemini-1.5-turbo")

    OLLAMA_API_KEY: str = os.getenv("OLLAMA_API_KEY", "").strip()
    OLLAMA_MODEL: str = os.getenv("OLLAMA_MODEL", "llama3")

    # --- Embeddings ---
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # --- Vector store ---
    CHROMA_PERSIST_DIR: str = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
    CHROMA_COLLECTION_NAME: str = "knowledge_base"

    # --- Chunking ---
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "500"))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "50"))

    # --- Retrieval ---
    RETRIEVAL_TOP_K: int = int(os.getenv("RETRIEVAL_TOP_K", "4"))
    # Chroma returns cosine *distance* (0 = identical, 2 = opposite).
    # We convert distance -> similarity in retrieval.py, and this threshold
    # is applied to that similarity score (0-1, higher = more relevant).
    RETRIEVAL_SCORE_THRESHOLD: float = float(
        os.getenv("RETRIEVAL_SCORE_THRESHOLD", "0.3")
    )

    # --- Memory ---
    MEMORY_MAX_TURNS: int = int(os.getenv("MEMORY_MAX_TURNS", "10"))

    # --- Paths ---
    DATA_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
    ORDERS_FILE: str = os.path.join(DATA_DIR, "orders.json")
    PRODUCTS_FILE: str = os.path.join(DATA_DIR, "products.json")
    UPLOADS_DIR: str = os.path.join(
        os.path.dirname(os.path.dirname(__file__)), "uploads"
    )


settings = Settings()

# Make sure runtime directories exist
os.makedirs(settings.UPLOADS_DIR, exist_ok=True)
os.makedirs(settings.CHROMA_PERSIST_DIR, exist_ok=True)