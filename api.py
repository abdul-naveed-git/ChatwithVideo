"""
FastAPI application powered by LangChain and ChromaDB.
No custom classes or dataclasses.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from config import RESPONSE_LANGUAGES
from generation.answer import ask_youtube
from ingestion.transcripts import (
    list_captions,
    load_youtube_transcript_documents,
)
from retrieval.store import add_documents_to_store, delete_by_video_id, video_is_indexed

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(name)-25s  %(levelname)-7s  %(message)s",
)
logger = logging.getLogger("api")

app = FastAPI(
    title="LangChain YouTube Multilingual RAG API",
    description="FastAPI service powered by LangChain, ChromaDB, multilingual E5, and Google Gemini.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ───────────────────────────────────

class IngestRequest(BaseModel):
    url: str = Field(description="YouTube video or playlist URL")
    language_code: str | None = Field(
        default=None,
        description="Caption language code. Defaults to English; use /captions/{video_id} to see other available languages.",
    )
    reingest: bool = Field(default=False, description="If True, overwrite existing chunks")


class ChatRequest(BaseModel):
    question: str = Field(description="Question in any language")
    response_language: str = Field(default="English", description="Target answer language")
    video_ids: list[str] | None = Field(default=None, description="Filter scope to specific video IDs")
    k: int = Field(default=5, description="Number of child chunks to retrieve")


# ── Endpoints ─────────────────────────────────

@app.get("/", tags=["Health"])
def root() -> dict[str, Any]:
    return {
        "service": "LangChain YouTube Multilingual RAG API",
        "framework": "LangChain LCEL",
        "vectorstore": "ChromaDB",
        "embeddings": "intfloat/multilingual-e5-small",
        "llm": "Google Gemini (ChatGoogleGenerativeAI)",
        "endpoints": ["/docs", "/health", "/ingest", "/chat", "/captions/{video_id}"],
    }


@app.get("/health", tags=["Health"])
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/supported-languages", tags=["Metadata"])
def supported_languages() -> list[str]:
    return RESPONSE_LANGUAGES


@app.get("/captions/{video_id}", tags=["Captions"])
def get_captions(video_id: str) -> list[dict[str, Any]]:
    """List available caption tracks for a video."""
    return list_captions(video_id)


@app.delete("/videos/{video_id}", tags=["Videos"])
def delete_video(video_id: str) -> dict[str, str]:
    """Delete chunks for a video ID from Chroma."""
    delete_by_video_id(video_id)
    return {"message": f"Deleted chunks for video {video_id}"}


@app.post("/ingest", tags=["Ingestion"])
def ingest(req: IngestRequest) -> dict[str, Any]:
    """Ingest YouTube video/playlist: Load -> Split -> Embed & Store in Chroma."""
    try:
        # 1. Load transcripts as LangChain Documents
        raw_docs = load_youtube_transcript_documents(
            url_or_video_id=req.url,
            language_code=req.language_code,
        )
        if not raw_docs:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No captions found or transcript is empty.",
            )

        # Validate captions before changing an existing index. Normal ingestion
        # reuses videos already present; reingest deliberately replaces them.
        video_ids = {doc.metadata.get("video_id") for doc in raw_docs}
        to_index = raw_docs if req.reingest else [
            doc for doc in raw_docs if not video_is_indexed(doc.metadata["video_id"])
        ]
        if to_index:
            add_documents_to_store(to_index)

        return {
            "status": "success",
            "raw_segments_loaded": len(raw_docs),
            "videos_processed": len(video_ids),
            "message": ("ParentDocumentRetriever refreshed indexed child chunks."
                        if req.reingest else "Existing lectures were reused; new lectures were indexed."),
        }
    except HTTPException:
        raise
    except Exception as exc:
        logger.error("Ingest failed: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))


@app.post("/chat", tags=["RAG Chain"])
def chat(req: ChatRequest) -> dict[str, Any]:
    """Execute LangChain RAG chain: Retriever | Prompt | ChatGoogleGenerativeAI | StrOutputParser."""
    try:
        result = ask_youtube(
            question=req.question,
            video_ids=req.video_ids,
            response_language=req.response_language,
            k=req.k,
        )
        return result
    except Exception as exc:
        logger.error("Chat error: %s", exc)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc))
