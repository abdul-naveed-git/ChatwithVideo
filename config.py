"""
Global configuration for LangChain YouTube RAG.
"""

from pathlib import Path

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parent
DATA_DIR: Path = PROJECT_ROOT / "data"
CHROMA_DIR: Path = DATA_DIR / "chroma"
PARENT_STORE_DIR: Path = DATA_DIR / "parent_documents_v2"

CHROMA_DIR.mkdir(parents=True, exist_ok=True)
PARENT_STORE_DIR.mkdir(parents=True, exist_ok=True)

# Models
EMBEDDING_MODEL_NAME: str = "intfloat/multilingual-e5-small"
CHROMA_COLLECTION_NAME: str = "video_parents_e5_v2"
GEMINI_MODEL_NAME: str = "gemini-3.5-flash-lite"

# Parent-child chunking. Values are character limits used by LangChain's
# recursive splitter, which works without adding a tokenizer runtime.
PARENT_CHUNK_SIZE = 2000
PARENT_CHUNK_OVERLAP = 200
CHILD_CHUNK_SIZE = 600
CHILD_CHUNK_OVERLAP = 100

RETRIEVAL_K = 10

# Backwards-compatible aliases for callers using the previous API.
CHUNK_SIZE: int = CHILD_CHUNK_SIZE
CHUNK_OVERLAP: int = CHILD_CHUNK_OVERLAP

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
