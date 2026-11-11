"""
LangChain Document chunking using RecursiveCharacterTextSplitter.
No custom classes or dataclasses.
"""

from __future__ import annotations

import re
try:
    from langchain_core.documents import Document
except ImportError:
    class Document:  # type: ignore[no-redef]
        def __init__(self, page_content: str, metadata: dict | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    from langchain_text_splitters import RecursiveCharacterTextSplitter
except ImportError:
    try:
        from langchain.text_splitter import RecursiveCharacterTextSplitter
    except ImportError:
        class RecursiveCharacterTextSplitter:  # type: ignore[no-redef]
            def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50, separators: list | None = None) -> None:
                self.chunk_size = chunk_size
                self.chunk_overlap = chunk_overlap
            def split_documents(self, documents: list[Document]) -> list[Document]:
                chunks = []
                for doc in documents:
                    text = doc.page_content
                    step = max(1, self.chunk_size - self.chunk_overlap)
                    for i in range(0, max(1, len(text)), step):
                        slice_text = text[i:i + self.chunk_size]
                        if slice_text:
                            chunks.append(Document(page_content=slice_text, metadata=dict(doc.metadata)))
                return chunks

from config import CHUNK_OVERLAP, CHUNK_SIZE

_TAG_RE = re.compile(r"\[.*?\]|\(.*?\)")


def clean_text(text: str) -> str:
    """Strip [Music] tags and excessive whitespace."""
    text = _TAG_RE.sub("", text)
    return " ".join(text.split())


def split_transcript_documents(
    documents: list[Document],
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> list[Document]:
    """Split LangChain Document list using RecursiveCharacterTextSplitter.

    Combines consecutive small caption segments per video, cleans noise,
    and applies RecursiveCharacterTextSplitter while preserving source metadata.
    """
    if not documents:
        return []

    # Clean document content
    cleaned_docs = []
    for doc in documents:
        cleaned = clean_text(doc.page_content)
        if cleaned:
            cleaned_docs.append(
                Document(
                    page_content=cleaned,
                    metadata=dict(doc.metadata),
                )
            )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = text_splitter.split_documents(cleaned_docs)
    return chunks
