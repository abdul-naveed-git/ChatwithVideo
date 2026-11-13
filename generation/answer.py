"""
LangChain RAG chain using LCEL, ChatGoogleGenerativeAI, ChatPromptTemplate, and StrOutputParser.
No custom classes or dataclasses.
"""

from __future__ import annotations

import logging
import os
from operator import itemgetter
from typing import Any

try:
    from langchain_core.documents import Document
    from langchain_core.output_parsers import StrOutputParser
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
except ImportError:
    class Document:  # type: ignore[no-redef]
        def __init__(self, page_content: str, metadata: dict | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}
    StrOutputParser = Any  # type: ignore[assignment, misc]
    ChatPromptTemplate = Any  # type: ignore[assignment, misc]
    ChatGoogleGenerativeAI = Any  # type: ignore[assignment, misc]

from config import GEMINI_MODEL_NAME
from retrieval.store import get_parent_retriever

logger = logging.getLogger(__name__)

SYSTEM_PROMPT_TEMPLATE = """You are a helpful assistant answering questions about YouTube videos using ONLY the transcript excerpts provided below.

Rules:
1. Use ONLY the provided excerpts. Do not use outside knowledge.
2. Write your entire answer in {response_language}.
3. If the excerpts do not contain the answer, state clearly in {response_language} that the video(s) do not cover it.
4. Combine relevant sections when the answer spans them.
5. Always cite the video title and timestamp URL provided in each excerpt header.
6. Keep technical terms, code, and proper nouns in their original form.

Excerpts:
{context}

Question: {question}
"""


def format_docs(docs: list[Document]) -> str:
    """Format retrieved LangChain documents with timestamp citation headers."""
    if not docs:
        return "No relevant video excerpts found."

    formatted = []
    for i, doc in enumerate(docs, 1):
        title = doc.metadata.get("title", "Video")
        start = doc.metadata.get("start")
        url = doc.metadata.get("url", "")
        if start is None:
            timestamp = "timestamp unavailable"
        else:
            mins = int(float(start) // 60)
            secs = int(float(start) % 60)
            timestamp = f"{mins}:{secs:02d}"
        header = f"[Source {i}: {title} @ {timestamp} ({url})]"
        formatted.append(f"{header}\n{doc.page_content}")

    return "\n\n".join(formatted)


def get_llm(model_name: str = GEMINI_MODEL_NAME, temperature: float = 0.2) -> Any:
    """Initialize LangChain Google GenAI Chat model."""
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY environment variable is not set.")
    return ChatGoogleGenerativeAI(
        model=model_name,
        temperature=temperature,
        google_api_key=api_key,
    )


def create_rag_chain(
    retriever: Any,
    model_name: str = GEMINI_MODEL_NAME,
) -> Any:
    """Build pure LangChain LCEL RAG chain.

    Inputs: {"question": str, "response_language": str}
    Outputs: str (LLM generated answer)
    """
    llm = get_llm(model_name=model_name)
    prompt = ChatPromptTemplate.from_template(SYSTEM_PROMPT_TEMPLATE)

    chain = (
        {
            "context": itemgetter("question") | retriever | format_docs,
            "question": itemgetter("question"),
            "response_language": itemgetter("response_language"),
        }
        | prompt
        | llm
        | StrOutputParser()
    )
    return chain


def ask_youtube(
    question: str,
    video_ids: list[str] | None = None,
    response_language: str = "English",
    k: int = 5,
) -> dict[str, Any]:
    """Execute LangChain RAG pipeline for a user question.

    Returns dict with answer, retrieved sources, and raw documents.
    """
    if response_language.lower().startswith("same"):
        response_language = "the same language as the question"

    parent_retriever = get_parent_retriever(video_ids=video_ids, k=k)
    retrieved_docs = parent_retriever.invoke(question)
    chain = create_rag_chain(parent_retriever)
    answer = chain.invoke({
        "question": question,
        "response_language": response_language,
    })

    # 3. Format sources
    sources = []
    for d in retrieved_docs:
        sources.append({
            "title": d.metadata.get("title", ""),
            "video_id": d.metadata.get("video_id", ""),
            "start": d.metadata.get("start"),
            "end": d.metadata.get("end_time"),
            "url": d.metadata.get("url", ""),
            "text": d.page_content,
        })

    return {
        "question": question,
        "answer": answer,
        "sources": sources,
        "documents": retrieved_docs,
    }
