"""
SQLite database for video-level metadata and chat sessions.

Tables
------
videos
    One row per ingested (or failed) video.
chat_sessions
    Conversation history for the Streamlit chat UI.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Generator

from config import SQLITE_DB_PATH

logger = logging.getLogger(__name__)

# ──────────────────────────────────────────────
# Schema
# ──────────────────────────────────────────────

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS videos (
    video_id        TEXT PRIMARY KEY,
    title           TEXT NOT NULL,
    playlist_id     TEXT,          -- NULL if added as a single video
    language        TEXT NOT NULL,  -- caption language code
    transcript_src  TEXT NOT NULL,  -- 'manual' | 'auto-generated'
    status          TEXT NOT NULL DEFAULT 'ingested',  -- 'ingested' | 'failed'
    fail_reason     TEXT,
    chunk_count     INTEGER DEFAULT 0,
    ingested_at     TEXT NOT NULL,  -- ISO-8601 UTC
    updated_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS chat_sessions (
    session_id  TEXT PRIMARY KEY,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL,
    history     TEXT NOT NULL DEFAULT '[]'   -- JSON array of {role, content}
);
"""


# ──────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────

@dataclass
class VideoRecord:
    """Mirror of a row in the ``videos`` table."""

    video_id: str
    title: str
    language: str
    transcript_src: str
    status: str = "ingested"
    playlist_id: str | None = None
    fail_reason: str | None = None
    chunk_count: int = 0
    ingested_at: str = ""
    updated_at: str = ""


@dataclass
class ChatMessage:
    """A single chat turn."""

    role: str       # "user" | "assistant"
    content: str


@dataclass
class ChatSession:
    """A chat session with serialised history."""

    session_id: str
    history: list[ChatMessage] = field(default_factory=list)
    created_at: str = ""
    updated_at: str = ""


# ──────────────────────────────────────────────
# Database manager
# ──────────────────────────────────────────────

class Database:
    """Thin SQLite wrapper for the RAG app."""

    def __init__(self, db_path: Path = SQLITE_DB_PATH) -> None:
        self.db_path = db_path
        self._init_db()

    # ── connection helpers ────────────────────

    @contextmanager
    def _conn(self) -> Generator[sqlite3.Connection, None, None]:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Create tables if they don't exist."""
        with self._conn() as conn:
            conn.executescript(_SCHEMA)
        logger.info("SQLite database ready at %s", self.db_path)

    # ── video CRUD ────────────────────────────

    def upsert_video(self, rec: VideoRecord) -> None:
        """Insert or update a video record."""
        now = _now_iso()
        rec.updated_at = now
        if not rec.ingested_at:
            rec.ingested_at = now
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO videos
                    (video_id, title, playlist_id, language, transcript_src,
                     status, fail_reason, chunk_count, ingested_at, updated_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(video_id) DO UPDATE SET
                    title          = excluded.title,
                    playlist_id    = excluded.playlist_id,
                    language       = excluded.language,
                    transcript_src = excluded.transcript_src,
                    status         = excluded.status,
                    fail_reason    = excluded.fail_reason,
                    chunk_count    = excluded.chunk_count,
                    updated_at     = excluded.updated_at
                """,
                (
                    rec.video_id, rec.title, rec.playlist_id, rec.language,
                    rec.transcript_src, rec.status, rec.fail_reason,
                    rec.chunk_count, rec.ingested_at, rec.updated_at,
                ),
            )

    def mark_failed(self, video_id: str, title: str, reason: str,
                    playlist_id: str | None = None) -> None:
        """Record a video that could not be ingested."""
        rec = VideoRecord(
            video_id=video_id,
            title=title,
            language="",
            transcript_src="",
            status="failed",
            playlist_id=playlist_id,
            fail_reason=reason,
        )
        self.upsert_video(rec)

    def get_video(self, video_id: str) -> VideoRecord | None:
        """Fetch a single video record or None."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM videos WHERE video_id = ?", (video_id,)
            ).fetchone()
        return _row_to_video(row) if row else None

    def is_ingested(self, video_id: str) -> bool:
        """Return True if the video was successfully ingested."""
        rec = self.get_video(video_id)
        return rec is not None and rec.status == "ingested"

    def list_videos(self, status: str | None = None) -> list[VideoRecord]:
        """List videos, optionally filtered by status."""
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM videos WHERE status = ? ORDER BY ingested_at DESC",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM videos ORDER BY ingested_at DESC"
                ).fetchall()
        return [_row_to_video(r) for r in rows]

    def delete_video(self, video_id: str) -> None:
        """Delete a video record (for re-ingest)."""
        with self._conn() as conn:
            conn.execute("DELETE FROM videos WHERE video_id = ?", (video_id,))

    # ── chat session CRUD ─────────────────────

    def upsert_session(self, session: ChatSession) -> None:
        """Insert or update a chat session."""
        now = _now_iso()
        session.updated_at = now
        if not session.created_at:
            session.created_at = now
        history_json = json.dumps(
            [{"role": m.role, "content": m.content} for m in session.history]
        )
        with self._conn() as conn:
            conn.execute(
                """
                INSERT INTO chat_sessions (session_id, created_at, updated_at, history)
                VALUES (?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    updated_at = excluded.updated_at,
                    history    = excluded.history
                """,
                (session.session_id, session.created_at, session.updated_at,
                 history_json),
            )

    def get_session(self, session_id: str) -> ChatSession | None:
        """Load a chat session by id."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
        if not row:
            return None
        history_raw = json.loads(row["history"])
        return ChatSession(
            session_id=row["session_id"],
            created_at=row["created_at"],
            updated_at=row["updated_at"],
            history=[ChatMessage(**m) for m in history_raw],
        )

    def list_sessions(self) -> list[ChatSession]:
        """List all chat sessions (without full history for speed)."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT session_id, created_at, updated_at FROM chat_sessions "
                "ORDER BY updated_at DESC"
            ).fetchall()
        return [
            ChatSession(
                session_id=r["session_id"],
                created_at=r["created_at"],
                updated_at=r["updated_at"],
            )
            for r in rows
        ]

    def delete_session(self, session_id: str) -> None:
        """Delete a chat session."""
        with self._conn() as conn:
            conn.execute(
                "DELETE FROM chat_sessions WHERE session_id = ?",
                (session_id,),
            )


# ──────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────

def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_video(row: sqlite3.Row) -> VideoRecord:
    return VideoRecord(
        video_id=row["video_id"],
        title=row["title"],
        playlist_id=row["playlist_id"],
        language=row["language"],
        transcript_src=row["transcript_src"],
        status=row["status"],
        fail_reason=row["fail_reason"],
        chunk_count=row["chunk_count"],
        ingested_at=row["ingested_at"],
        updated_at=row["updated_at"],
    )


# ──────────────────────────────────────────────
# Module-level singleton
# ──────────────────────────────────────────────

_db_instance: Database | None = None


def get_db() -> Database:
    """Return a lazily-initialised Database singleton."""
    global _db_instance
    if _db_instance is None:
        _db_instance = Database()
    return _db_instance
