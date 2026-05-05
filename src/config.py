# filepath: src/config.py
import os
from dotenv import load_dotenv

load_dotenv()

# ── LangSmith Observability ────────────────────────────────────
# Must be set before any langchain import takes effect.
# These are read automatically by LangChain/LangGraph internals.
os.environ["LANGCHAIN_TRACING_V2"] = os.getenv("LANGCHAIN_TRACING_V2", "true")
os.environ["LANGCHAIN_API_KEY"]    = os.getenv("LANGCHAIN_API_KEY", "")
os.environ["LANGCHAIN_PROJECT"]    = os.getenv("LANGCHAIN_PROJECT", "agriculture-advisor-ai407l")


class Config:
    """
    Configuration class for environment variables and paths.
    Uses os.getenv for flexibility in deployment.
    """

    # ── Groq LLM ──────────────────────────────────────────────
    GROQ_API_KEY: str = os.getenv("GROQ_API_KEY")
    MODEL_NAME: str   = os.getenv("MODEL_NAME", "llama-3.1-8b-instant")

    # ── Embeddings ────────────────────────────────────────────
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    # ── Persistence ───────────────────────────────────────────
    CHECKPOINT_DB_PATH: str = os.getenv("CHECKPOINT_DB_PATH", "checkpoint_db.sqlite")
    DATABASE_URL: str      = os.getenv("DATABASE_URL", "") # For Postgres in Docker
    PERSISTENCE_MODE: str  = os.getenv("PERSISTENCE_MODE", "sqlite") # 'sqlite' or 'postgres'

    # ── Data Paths ────────────────────────────────────────────
    PDF_DATA_PATH: str       = os.path.join("data", "raw_pdfs")
    EMBEDDINGS_DATA_PATH: str = os.path.join("data", "embeddings")
    FAISS_INDEX_NAME: str    = "agriculture_index"

    # ── LangSmith Observability ───────────────────────────────
    LANGCHAIN_TRACING_V2: str = os.getenv("LANGCHAIN_TRACING_V2", "true")
    LANGCHAIN_API_KEY: str    = os.getenv("LANGCHAIN_API_KEY", "")
    LANGCHAIN_PROJECT: str    = os.getenv("LANGCHAIN_PROJECT", "agriculture-advisor-ai407l")