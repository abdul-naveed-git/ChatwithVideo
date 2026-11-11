"""
LangChain Embeddings factory for BAAI/bge-m3.
No custom classes. Uses standard LangChain HuggingFaceEmbeddings.
"""

from __future__ import annotations

import logging
from typing import Any

from config import EMBEDDING_MODEL_NAME

logger = logging.getLogger(__name__)

_cached_embeddings = None


def get_embeddings(model_name: str = EMBEDDING_MODEL_NAME) -> Any:
    """Return LangChain HuggingFaceEmbeddings instance for BAAI/bge-m3."""
    global _cached_embeddings
    if _cached_embeddings is not None:
        return _cached_embeddings

    try:
        from langchain_huggingface import HuggingFaceEmbeddings
        logger.info("Initializing LangChain HuggingFaceEmbeddings: %s", model_name)
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            encode_kwargs={"normalize_embeddings": True},
        )
        return _cached_embeddings
    except ImportError:
        try:
            from langchain_community.embeddings import HuggingFaceEmbeddings
            logger.info("Initializing LangChain Community HuggingFaceEmbeddings: %s", model_name)
            _cached_embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                encode_kwargs={"normalize_embeddings": True},
            )
            return _cached_embeddings
        except ImportError as exc:
            raise ImportError(
                "Please install langchain-huggingface or sentence-transformers:\n"
                "  pip install langchain-huggingface sentence-transformers"
            ) from exc
