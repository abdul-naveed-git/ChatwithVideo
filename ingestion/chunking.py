"""Transcript assembly and LangChain splitters for caption-based videos."""
from __future__ import annotations

import re
from collections import defaultdict
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

_NOISE = re.compile(r"\s*[\[(](?:music|applause|laughter|inaudible|silence)[\])]\s*", re.I)

def clean_text(text: str) -> str:
    """Remove only known caption cues; retain meaningful parentheticals such as ``(ML)``."""
    return re.sub(r"\s+", " ", _NOISE.sub(" ", text or "")).strip()

def assemble_transcript_documents(documents: list[Document]) -> list[Document]:
    """Create one ordered transcript per video, retaining caption character spans."""
    grouped: dict[str, list[Document]] = defaultdict(list)
    for doc in documents:
        video_id = doc.metadata.get("video_id")
        if not video_id:
            raise ValueError("Every document must contain video_id")
        if doc.metadata.get("start") is None:
            raise ValueError("Every document must contain a start timestamp")
        grouped[str(video_id)].append(doc)
    assembled = []
    for video_id, captions in grouped.items():
        captions.sort(key=lambda d: float(d.metadata["start"]))
        parts, spans, cursor = [], [], 0
        for caption in captions:
            text = clean_text(caption.page_content)
            if not text:
                continue
            if parts:
                parts.append("\n"); cursor += 1
            begin = cursor; parts.append(text); cursor += len(text)
            meta = caption.metadata; start = float(meta["start"]); duration = float(meta.get("duration") or 0.0)
            spans.append({"start_char": begin, "end_char": cursor, "start": start, "end": start + max(0.0, duration)})
        if spans:
            metadata = dict(captions[0].metadata)
            metadata.setdefault("title", f"Video {video_id}")
            metadata.setdefault("language", "unknown")
            metadata.setdefault("url", f"https://youtu.be/{video_id}?t={int(spans[0]['start'])}")
            metadata.update({"video_id": video_id, "caption_spans": spans, "start": spans[0]["start"],
                             "end_time": spans[-1]["end"], "citation_precision": "caption-backed range"})
            assembled.append(Document(page_content="".join(parts), metadata=metadata))
    return assembled

class TimestampedParentSplitter(RecursiveCharacterTextSplitter):
    """LangChain splitter which maps parent character ranges to source caption times."""
    def split_documents(self, documents: list[Document]) -> list[Document]:
        chunks = super().split_documents(documents)
        for chunk in chunks:
            offset = int(chunk.metadata.get("start_index", 0)); end_offset = offset + len(chunk.page_content)
            covered = [s for s in chunk.metadata.get("caption_spans", [])
                       if s["end_char"] > offset and s["start_char"] < end_offset]
            if covered:
                chunk.metadata["start"] = min(s["start"] for s in covered)
                chunk.metadata["start_time"] = chunk.metadata["start"]
                chunk.metadata["end_time"] = max(s["end"] for s in covered)
                chunk.metadata["citation_precision"] = "caption-backed range"
                chunk.metadata["url"] = f"https://youtu.be/{chunk.metadata['video_id']}?t={int(chunk.metadata['start'])}"
            else:
                chunk.metadata.update({"start": None, "start_time": None, "end_time": None, "citation_precision": "unavailable"})
        return chunks
