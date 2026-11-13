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
from ingestion.transcripts import (
    _choose_caption_language,
    extract_playlist_id,
    extract_video_id,
)
from ingestion.chunking import TimestampedParentSplitter, assemble_transcript_documents, clean_text
from retrieval.embedder import E5Embeddings
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


class TestCaptionLanguageSelection(unittest.TestCase):
    def test_defaults_to_english_even_when_other_manual_caption_is_first(self):
        available = [
            {"language_code": "hi", "is_generated": False},
            {"language_code": "en", "is_generated": True},
        ]
        self.assertEqual(_choose_caption_language(available, None), "en")

    def test_uses_requested_non_english_language(self):
        available = [{"language_code": "en", "is_generated": False}]
        self.assertEqual(_choose_caption_language(available, "te"), "te")

    def test_falls_back_when_english_is_unavailable(self):
        available = [
            {"language_code": "hi", "is_generated": True},
            {"language_code": "ta", "is_generated": False},
        ]
        self.assertEqual(_choose_caption_language(available, None), "ta")


class TestLangChainChunking(unittest.TestCase):
    def test_clean_text(self):
        self.assertEqual(clean_text("[Music] Hello world"), "Hello world")
        self.assertEqual(clean_text("(Applause) Thank you!"), "Thank you!")
        self.assertEqual(clean_text("Line 1\n   Line 2"), "Line 1 Line 2")

    def test_assemble_transcript_documents(self):
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
        transcripts = assemble_transcript_documents(raw_docs)
        self.assertEqual(len(transcripts), 1)
        self.assertNotIn("[Music]", transcripts[0].page_content)
        self.assertEqual(transcripts[0].metadata["video_id"], "vid1")

    def test_parent_splitting_preserves_caption_timestamp_ranges(self):
        raw_docs = [
            Document(
                page_content=f"Caption {i} explains a part of the process.",
                metadata={"video_id": "vid1", "title": "Demo", "start": float(i * 10),
                          "duration": 8.0, "url": "https://youtu.be/vid1"},
            )
            for i in range(8)
        ]
        parents = TimestampedParentSplitter(
            chunk_size=80, chunk_overlap=10, add_start_index=True,
        ).split_documents(assemble_transcript_documents(raw_docs))
        self.assertGreater(len(parents), 1)
        for parent in parents:
            self.assertLessEqual(parent.metadata["start_time"], parent.metadata["end_time"])


class TestE5Embeddings(unittest.TestCase):
    def test_e5_uses_document_and_query_prefixes(self):
        class FakeEmbeddings:
            def __init__(self):
                self.documents = []
                self.query = ""
            def embed_documents(self, texts):
                self.documents = texts
                return [[1.0]]
            def embed_query(self, text):
                self.query = text
                return [1.0]

        fake = FakeEmbeddings()
        embeddings = E5Embeddings(fake)
        embeddings.embed_documents(["hola"])
        embeddings.embed_query("what is this?")
        self.assertEqual(fake.documents, ["passage: hola"])
        self.assertEqual(fake.query, "query: what is this?")
        embeddings.embed_documents(["passage: hola"])
        embeddings.embed_query("query: what is this?")
        self.assertEqual(fake.documents, ["passage: hola"])
        self.assertEqual(fake.query, "query: what is this?")


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
