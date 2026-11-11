"""
LangChain Chroma vector store functions.
No custom classes.
"""

from __future__ import annotations

import logging
from typing import Any

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:  # type: ignore[no-redef]
        def __init__(self, page_content: str, metadata: dict | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

from config import CHROMA_COLLECTION_NAME, CHROMA_DIR
from retrieval.embedder import get_embeddings

logger = logging.getLogger(__name__)

_cached_store = None


def get_vectorstore(
    collection_name: str = CHROMA_COLLECTION_NAME,
    persist_directory: str = str(CHROMA_DIR),
    embedding_function: Any = None,
) -> Any:
    """Return LangChain Chroma vector store instance."""
    global _cached_store
    if _cached_store is not None:
        return _cached_store

    embeddings = embedding_function or get_embeddings()

    try:
        from langchain_chroma import Chroma
        _cached_store = Chroma(
            collection_name=collection_name,
            embedding_function=embeddings,
            persist_directory=persist_directory,
        )
        return _cached_store
    except ImportError:
        try:
            from langchain_community.vectorstores import Chroma
            _cached_store = Chroma(
                collection_name=collection_name,
                embedding_function=embeddings,
                persist_directory=persist_directory,
            )
            return _cached_store
        except ImportError as exc:
            raise ImportError(
                "Please install langchain-chroma or chromadb:\n"
                "  pip install langchain-chroma chromadb"
            ) from exc


def add_documents_to_store(documents: list[Document]) -> list[str]:
    """Embed and insert LangChain Document list into Chroma."""
    if not documents:
        return []
    store = get_vectorstore()
    ids = store.add_documents(documents)
    logger.info("Added %d documents to Chroma collection '%s'", len(documents), CHROMA_COLLECTION_NAME)
    return ids


def get_retriever(
    video_ids: list[str] | None = None,
    k: int = 5,
) -> Any:
    """Return LangChain retriever with optional video_id metadata filter."""
    store = get_vectorstore()

    search_kwargs: dict[str, Any] = {"k": k}
    if video_ids:
        if len(video_ids) == 1:
            search_kwargs["filter"] = {"video_id": video_ids[0]}
        else:
            search_kwargs["filter"] = {"video_id": {"$in": video_ids}}

    return store.as_retriever(search_kwargs=search_kwargs)


def delete_by_video_id(video_id: str) -> None:
    """Delete chunks for a specific video ID from Chroma."""
    store = get_vectorstore()
    try:
        store.delete(where={"video_id": video_id})
        logger.info("Deleted chunks for video %s from Chroma", video_id)
    except Exception as exc:
        logger.warning("Could not delete chunks for %s: %s", video_id, exc)
