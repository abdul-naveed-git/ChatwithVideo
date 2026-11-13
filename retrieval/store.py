"""Persistent Chroma + LangChain ParentDocumentRetriever storage."""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Any
from langchain_core.documents import Document
from langchain_classic.retrievers import ParentDocumentRetriever
from langchain_classic.storage import LocalFileStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config import CHILD_CHUNK_OVERLAP, CHILD_CHUNK_SIZE, CHROMA_COLLECTION_NAME, CHROMA_DIR, PARENT_CHUNK_OVERLAP, PARENT_CHUNK_SIZE, PARENT_STORE_DIR, RETRIEVAL_K
from ingestion.chunking import TimestampedParentSplitter, assemble_transcript_documents
from retrieval.embedder import get_embeddings
logger = logging.getLogger(__name__)
_cached_stores: dict[tuple[str, str, int], Any] = {}

class CitationParentDocumentRetriever(ParentDocumentRetriever):
    """Expose the real persistent parent key in returned parent metadata."""
    def _split_docs_for_adding(self, *args: Any, **kwargs: Any):
        children, parents = super()._split_docs_for_adding(*args, **kwargs)
        for parent_id, parent in parents:
            parent.metadata["parent_id"] = parent_id
        return children, parents

def get_vectorstore(collection_name: str = CHROMA_COLLECTION_NAME, persist_directory: str | Path = CHROMA_DIR, embedding_function: Any = None) -> Any:
    from langchain_chroma import Chroma
    embeddings = embedding_function or get_embeddings(); key = (collection_name, str(Path(persist_directory).resolve()), id(embeddings))
    if key not in _cached_stores:
        _cached_stores[key] = Chroma(collection_name=collection_name, embedding_function=embeddings, persist_directory=str(persist_directory))
    return _cached_stores[key]

def get_parent_retriever(video_ids: list[str] | None = None, k: int = RETRIEVAL_K, *, collection_name: str = CHROMA_COLLECTION_NAME, persist_directory: str | Path = CHROMA_DIR, parent_store_directory: str | Path = PARENT_STORE_DIR, embedding_function: Any = None) -> Any:
    search_kwargs: dict[str, Any] = {"k": k}
    if video_ids: search_kwargs["filter"] = {"video_id": video_ids[0]} if len(video_ids) == 1 else {"video_id": {"$in": video_ids}}
    return CitationParentDocumentRetriever(vectorstore=get_vectorstore(collection_name, persist_directory, embedding_function), byte_store=LocalFileStore(parent_store_directory), id_key="parent_id", search_kwargs=search_kwargs,
        parent_splitter=TimestampedParentSplitter(chunk_size=PARENT_CHUNK_SIZE, chunk_overlap=PARENT_CHUNK_OVERLAP, add_start_index=True, separators=["\n\n", "\n", ". ", " ", ""]),
        child_splitter=RecursiveCharacterTextSplitter(chunk_size=CHILD_CHUNK_SIZE, chunk_overlap=CHILD_CHUNK_OVERLAP, separators=["\n\n", "\n", ". ", " ", ""]),
        # Chroma child metadata must remain scalar; caption_spans stay on parents.
        child_metadata_fields=["video_id", "title", "language", "start", "start_time", "end_time", "url", "citation_precision"])

def add_documents_to_store(documents: list[Document], **store_options: Any) -> None:
    """Accept loader captions, assemble by video, then let LangChain create parents and children."""
    if not documents: return
    assembled = assemble_transcript_documents(documents)
    for video_id in {d.metadata["video_id"] for d in assembled}: delete_by_video_id(video_id, **store_options)
    get_parent_retriever(**store_options).add_documents(assembled)
    logger.info("Assembled %d captions into %d video transcripts", len(documents), len(assembled))

add_parent_documents_to_store = add_documents_to_store
def retrieve_parent_documents(question: str, video_ids: list[str] | None = None, k: int = RETRIEVAL_K, **opts: Any) -> list[Document]: return get_parent_retriever(video_ids, k, **opts).invoke(question)
def get_retriever(k: int = RETRIEVAL_K, video_ids: list[str] | None = None, **opts: Any) -> Any: return get_parent_retriever(video_ids, k, **opts)
def video_is_indexed(video_id: str, **store_options: Any) -> bool:
    """Return whether this collection already has child vectors for a video."""
    collection = get_parent_retriever(**store_options).vectorstore._collection
    result = collection.get(where={"video_id": video_id}, limit=1, include=[])
    return bool(result.get("ids"))
def delete_by_video_id(video_id: str, **store_options: Any) -> None:
    retriever = get_parent_retriever(**store_options); retriever.vectorstore.delete(where={"video_id": video_id})
    keys = list(retriever.docstore.yield_keys()); docs = retriever.docstore.mget(keys)
    doomed = [key for key, doc in zip(keys, docs) if doc and doc.metadata.get("video_id") == video_id]
    if doomed: retriever.docstore.mdelete(doomed)
    logger.info("Deleted %d parents and matching child vectors for %s", len(doomed), video_id)
