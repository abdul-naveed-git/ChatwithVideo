"""
YouTube transcript loader returning LangChain Document objects.
"""

from __future__ import annotations

import logging
import re
from typing import Any

try:
    from langchain_core.documents import Document
except ImportError:
    class Document:  # type: ignore[no-redef]
        def __init__(self, page_content: str, metadata: dict | None = None) -> None:
            self.page_content = page_content
            self.metadata = metadata or {}

try:
    import yt_dlp
except ImportError:
    yt_dlp = None

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    from youtube_transcript_api._errors import (
        NoTranscriptFound,
        TranscriptsDisabled,
        VideoUnavailable,
    )
except ImportError:
    YouTubeTranscriptApi = None
    NoTranscriptFound = Exception
    TranscriptsDisabled = Exception
    VideoUnavailable = Exception

logger = logging.getLogger(__name__)

_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/v/|/shorts/)([a-zA-Z0-9_-]{11})")
_PLAYLIST_ID_RE = re.compile(r"[?&]list=([a-zA-Z0-9_-]+)")


def extract_video_id(url: str) -> str | None:
    """Extract 11-char YouTube video ID from URL or return string if already ID."""
    if len(url) == 11 and not ("/" in url or "?" in url):
        return url
    m = _VIDEO_ID_RE.search(url)
    return m.group(1) if m else None


def extract_playlist_id(url: str) -> str | None:
    """Extract playlist ID if present."""
    m = _PLAYLIST_ID_RE.search(url)
    return m.group(1) if m else None


def fetch_video_metadata(url: str) -> list[dict[str, str]]:
    """Return list of dicts: [{'video_id': ..., 'title': ...}]."""
    if yt_dlp is None:
        vid = extract_video_id(url)
        return [{"video_id": vid or url, "title": f"Video {vid or url}"}]

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "skip_download": True,
        "extract_flat": True,
        "ignoreerrors": True,
    }
    results = []
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=False)
        if not info:
            vid = extract_video_id(url)
            return [{"video_id": vid or url, "title": f"Video {vid or url}"}]

        entries = info.get("entries")
        if entries:
            for entry in entries:
                if entry:
                    vid = entry.get("id") or entry.get("url", "")
                    title = entry.get("title") or vid
                    results.append({"video_id": vid, "title": title})
        else:
            vid = info.get("id") or extract_video_id(url) or ""
            title = info.get("title") or vid
            results.append({"video_id": vid, "title": title})

    return results


def list_captions(video_id: str) -> list[dict[str, Any]]:
    """Return available caption tracks."""
    if YouTubeTranscriptApi is None:
        return []
    try:
        api = YouTubeTranscriptApi()
        tlist = api.list(video_id)
        return [
            {
                "language": t.language,
                "language_code": t.language_code,
                "is_generated": t.is_generated,
            }
            for t in tlist
        ]
    except Exception as exc:
        logger.warning("Could not list captions for %s: %s", video_id, exc)
        return []


def load_youtube_transcript_documents(
    url_or_video_id: str,
    language_code: str | None = None,
) -> list[Document]:
    """Fetch transcripts and return them directly as LangChain Document objects."""
    if YouTubeTranscriptApi is None:
        raise ImportError("Please install youtube-transcript-api: pip install youtube-transcript-api")

    metas = fetch_video_metadata(url_or_video_id)
    all_docs: list[Document] = []
    api = YouTubeTranscriptApi()

    for meta in metas:
        vid = meta["video_id"]
        title = meta["title"]

        try:
            available = list_captions(vid)
            if not available:
                logger.warning("No captions available for %s", vid)
                continue

            # Pick requested language or prefer manual captions
            target_lang = language_code
            if not target_lang:
                manual = [a["language_code"] for a in available if not a["is_generated"]]
                target_lang = manual[0] if manual else available[0]["language_code"]

            fetched = api.fetch(vid, languages=[target_lang])

            # Group caption snippets into LangChain Document blocks
            for snippet in fetched:
                start_sec = snippet.start
                url = f"https://youtu.be/{vid}?t={max(0, int(start_sec) - 5)}"
                doc = Document(
                    page_content=snippet.text.strip(),
                    metadata={
                        "video_id": vid,
                        "title": title,
                        "language": target_lang,
                        "start": start_sec,
                        "duration": snippet.duration,
                        "url": url,
                        "source": f"{title} [{int(start_sec)}s]",
                    },
                )
                all_docs.append(doc)

        except Exception as exc:
            logger.error("Failed to load transcript for %s: %s", vid, exc)

    return all_docs
