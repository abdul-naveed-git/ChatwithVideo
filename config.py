"""
Global configuration for LangChain YouTube RAG.
"""

from pathlib import Path

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHROMA_DIR: Path = DATA_DIR / "chroma"
SQLITE_DB_PATH: Path = DATA_DIR / "rag.db"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)

# Models
EMBEDDING_MODEL_NAME: str = "BAAI/bge-m3"
CHROMA_COLLECTION_NAME: str = "chunks__bge-m3"
GEMINI_MODEL_NAME: str = "gemini-3.5-flash-lite"

# Chunking
CHUNK_SIZE: int = 500
CHUNK_OVERLAP: int = 50

# Supported languages
RESPONSE_LANGUAGES: list[str] = [
    "Same as my question",
    "English",
    "Hindi",
    "Telugu",
    "Tamil",
    "Spanish",
    "French",
    "German",
    "Japanese",
    "Korean",
    "Portuguese",
    "Arabic",
    "Chinese (Simplified)",
]
