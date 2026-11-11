"""
Unit tests for LangChain YouTube RAG.
No complex runtimes or GPU required.
"""

import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from langchain_core.documents import Document
except ImportError:
    from ingestion.transcripts import Document
from ingestion.transcripts import extract_video_id, extract_playlist_id
from ingestion.chunking import clean_text, split_transcript_documents
from generation.answer import format_docs


class TestURLParsing(unittest.TestCase):
    def test_extract_video_id(self):
        urls = [
            ("https://www.youtube.com/watch?v=dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtu.be/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://youtube.com/embed/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("https://www.youtube.com/shorts/dQw4w9WgXcQ", "dQw4w9WgXcQ"),
            ("dQw4w9WgXcQ", "dQw4w9WgXcQ"),
        ]
        for url, expected in urls:
            self.assertEqual(extract_video_id(url), expected)

    def test_extract_playlist_id(self):
        self.assertEqual(
            extract_playlist_id("https://www.youtube.com/playlist?list=PL1234567890"),
            "PL1234567890",
        )
        self.assertIsNone(extract_playlist_id("https://youtu.be/dQw4w9WgXcQ"))


class TestLangChainChunking(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("[Music] Hello world"), "Hello world")
        self.assertEqual(clean_text("(Applause) Thank you!"), "Thank you!")
        self.assertEqual(clean_text("Line 1\n   Line 2"), "Line 1 Line 2")

    def test_split_transcript_documents(self):
        raw_docs = [
            Document(
                page_content="[Music] Artificial intelligence is transforming technology worldwide.",
                metadata={"video_id": "vid1", "title": "AI Video", "start": 0.0, "url": "https://youtu.be/vid1?t=0"},
            ),
            Document(
                page_content="Retrieval augmented generation connects external data to large language models.",
                metadata={"video_id": "vid1", "title": "AI Video", "start": 15.0, "url": "https://youtu.be/vid1?t=10"},
            ),
        ]
        chunks = split_transcript_documents(raw_docs, chunk_size=100, chunk_overlap=10)
        self.assertGreater(len(chunks), 0)
        self.assertNotIn("[Music]", chunks[0].page_content)
        self.assertEqual(chunks[0].metadata["video_id"], "vid1")


class TestFormattingAndCitations(unittest.TestCase):
    def test_format_docs(self):
        docs = [
            Document(
                page_content="First segment content",
                metadata={"title": "Deep Learning Tutorial", "start": 65.0, "url": "https://youtu.be/xyz?t=60"},
            )
        ]
        formatted = format_docs(docs)
        self.assertIn("Deep Learning Tutorial", formatted)
        self.assertIn("1:05", formatted)  # 65s -> 1:05
        self.assertIn("https://youtu.be/xyz?t=60", formatted)
        self.assertIn("First segment content", formatted)


if __name__ == "__main__":
    unittest.main()
