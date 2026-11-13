"""
YouTube transcript loader returning LangChain Document objects.
"""

from __future__ import annotations

import logging
import os
import re
import socket
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

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
except ImportError:
    YouTubeTranscriptApi = None

logger = logging.getLogger(__name__)

# English is the ingestion default.  A caller can still request any caption
# language returned by ``list_captions`` via ``language_code``.
DEFAULT_CAPTION_LANGUAGE = "en"
SUPADATA_TRANSCRIPT_URL = "https://api.supadata.ai/v1/transcript"
SUPADATA_TIMEOUT_SECONDS = 30

_VIDEO_ID_RE = re.compile(r"(?:v=|youtu\.be/|/embed/|/v/|/shorts/)([a-zA-Z0-9_-]{11})")
_PLAYLIST_ID_RE = re.compile(r"[?&]list=([a-zA-Z0-9_-]+)")


class SupadataTranscriptError(RuntimeError):
    """Base error for Supadata transcript failures; never includes the API key."""


class SupadataMissingCaptionsError(SupadataTranscriptError):
    """Supadata reports that the video has no native transcript."""


class SupadataCredentialsError(SupadataTranscriptError):
    """Supadata rejected the configured API key."""


class SupadataRateLimitError(SupadataTranscriptError):
    """Supadata rate or quota limit was reached."""


class SupadataTimeoutError(SupadataTranscriptError):
    """The Supadata request timed out or could not be reached."""


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


def normalize_youtube_url(url_or_video_id: str) -> str:
    """Return the canonical video URL accepted by Supadata."""
    video_id = extract_video_id(url_or_video_id)
    if not video_id:
        raise ValueError("Supadata ingestion requires a valid single YouTube video URL or video ID.")
    return f"https://www.youtube.com/watch?v={video_id}"


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


def _choose_caption_language(
    available: list[dict[str, Any]], language_code: str | None,
) -> str:
    """Select an explicitly requested caption language, else prefer English."""
    if language_code:
        return language_code

    # Prefer a manually supplied English track, then generated English.  Some
    # videos expose regional English tracks such as ``en-US`` rather than ``en``.
    english_tracks = [
        track for track in available
        if track["language_code"].lower() == DEFAULT_CAPTION_LANGUAGE
        or track["language_code"].lower().startswith(f"{DEFAULT_CAPTION_LANGUAGE}-")
    ]
    if english_tracks:
        manual_english = next(
            (track for track in english_tracks if not track["is_generated"]),
            None,
        )
        return (manual_english or english_tracks[0])["language_code"]

    # English is not available, so preserve the previous graceful fallback.
    manual = next((track for track in available if not track["is_generated"]), None)
    fallback = (manual or available[0])["language_code"]
    logger.info(
        "English captions are unavailable; using available caption language %s",
        fallback,
    )
    return fallback


def load_youtube_transcript_documents(
    url_or_video_id: str,
    language_code: str | None = None,
) -> list[Document]:
    """Fetch captions in English by default, or in ``language_code`` if specified."""
    supadata_api_key = os.environ.get("SUPADATA_API_KEY")
    if supadata_api_key:
        # This branch intentionally runs before yt-dlp/list_captions so a
        # direct YouTube IP block cannot prevent cloud-backed ingestion.
        return _load_supadata_transcript_documents(
            url_or_video_id, language_code, supadata_api_key,
        )

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

            target_lang = _choose_caption_language(available, language_code)

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


def _load_supadata_transcript_documents(
    url_or_video_id: str,
    language_code: str | None,
    api_key: str,
) -> list[Document]:
    """Fetch a native Supadata transcript and convert milliseconds to seconds."""
    normalized_url = normalize_youtube_url(url_or_video_id)
    video_id = extract_video_id(normalized_url)
    assert video_id is not None  # guaranteed by normalize_youtube_url
    params = {"url": normalized_url, "mode": "native", "text": "false"}
    if language_code:
        params["lang"] = language_code
    request = Request(
        f"{SUPADATA_TRANSCRIPT_URL}?{urlencode(params)}",
        headers={"x-api-key": api_key, "Accept": "application/json"},
        method="GET",
    )
    try:
        with urlopen(request, timeout=SUPADATA_TIMEOUT_SECONDS) as response:
            import json
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        if exc.code in (401, 403):
            raise SupadataCredentialsError("Supadata rejected SUPADATA_API_KEY.") from exc
        if exc.code == 429:
            raise SupadataRateLimitError("Supadata rate or quota limit reached; retry later.") from exc
        if exc.code in (404, 422):
            raise SupadataMissingCaptionsError("No native captions are available through Supadata.") from exc
        raise SupadataTranscriptError(f"Supadata transcript request failed (HTTP {exc.code}).") from exc
    except (TimeoutError, socket.timeout, URLError) as exc:
        raise SupadataTimeoutError("Supadata transcript request timed out or could not be reached.") from exc

    content = payload.get("content", payload.get("transcript", []))
    if not isinstance(content, list) or not content:
        error = str(payload.get("error", "")).lower()
        if "transcript-unavailable" in error or "caption" in error:
            raise SupadataMissingCaptionsError("No native captions are available through Supadata.")
        raise SupadataMissingCaptionsError("Supadata returned no transcript content.")

    metadata = payload.get("metadata") or {}
    title = (
        payload.get("title")
        or metadata.get("title")
        or f"Video {video_id}"
    )
    transcript_language = payload.get("lang") or language_code or DEFAULT_CAPTION_LANGUAGE
    documents = []
    for segment in content:
        text = str(segment.get("text", "")).strip()
        if not text:
            continue
        start = float(segment.get("offset", 0)) / 1000.0
        duration = float(segment.get("duration", 0)) / 1000.0
        documents.append(Document(
            page_content=text,
            metadata={
                "video_id": video_id,
                "title": title,
                "language": transcript_language,
                "start": start,
                "duration": duration,
                "url": f"https://youtu.be/{video_id}?t={max(0, int(start) - 5)}",
                "source": f"{title} [{int(start)}s]",
            },
        ))
    if not documents:
        raise SupadataMissingCaptionsError("Supadata transcript contains no caption text.")
    return documents
