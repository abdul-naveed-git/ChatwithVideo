"""
Unit tests for LangChain YouTube RAG.
No complex runtimes or GPU required.
"""

import sys
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

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
    load_youtube_transcript_documents,
    SupadataCredentialsError,
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


class TestTranscriptLoaders(unittest.TestCase):
    def test_supadata_loader_precedes_direct_youtube_and_converts_milliseconds(self):
        class Response:
            def read(self):
                return b'{"title":"Cloud lecture","lang":"te","content":[{"text":"First","offset":1250,"duration":750},{"text":"Second","offset":2000,"duration":500}]}'
            def __enter__(self): return self
            def __exit__(self, *args): return None

        with patch.dict("os.environ", {"SUPADATA_API_KEY": "test-key"}, clear=False), \
             patch("ingestion.transcripts.urlopen", return_value=Response()) as request, \
             patch("ingestion.transcripts.list_captions") as direct_listing:
            docs = load_youtube_transcript_documents("https://youtu.be/dQw4w9WgXcQ", "te")

        self.assertEqual([doc.metadata["start"] for doc in docs], [1.25, 2.0])
        self.assertEqual([doc.metadata["duration"] for doc in docs], [0.75, 0.5])
        self.assertEqual(docs[0].metadata["title"], "Cloud lecture")
        self.assertEqual(docs[0].metadata["language"], "te")
        direct_listing.assert_not_called()
        request_url = request.call_args.args[0].full_url
        self.assertIn("mode=native", request_url)
        self.assertIn("text=false", request_url)
        self.assertIn("lang=te", request_url)
        self.assertIn("youtube.com%2Fwatch%3Fv%3DdQw4w9WgXcQ", request_url)

    def test_supadata_invalid_credentials_are_distinct(self):
        from urllib.error import HTTPError
        error = HTTPError("https://api.supadata.ai/v1/transcript", 401, "Unauthorized", {}, None)
        with patch.dict("os.environ", {"SUPADATA_API_KEY": "test-key"}, clear=False), \
             patch("ingestion.transcripts.urlopen", side_effect=error):
            with self.assertRaises(SupadataCredentialsError):
                load_youtube_transcript_documents("dQw4w9WgXcQ")

    def test_direct_loader_is_used_without_supadata_key(self):
        class Snippet:
            text = "Direct caption"; start = 3.0; duration = 1.5
        api = Mock()
        api.fetch.return_value = [Snippet()]
        with patch.dict("os.environ", {}, clear=True), \
             patch("ingestion.transcripts.fetch_video_metadata", return_value=[{"video_id": "dQw4w9WgXcQ", "title": "Direct lecture"}]), \
             patch("ingestion.transcripts.list_captions", return_value=[{"language_code": "en", "is_generated": False}]), \
             patch("ingestion.transcripts.YouTubeTranscriptApi", return_value=api), \
             patch("ingestion.transcripts.urlopen") as supadata_request:
            docs = load_youtube_transcript_documents("dQw4w9WgXcQ")

        self.assertEqual(docs[0].page_content, "Direct caption")
        self.assertEqual(docs[0].metadata["start"], 3.0)
        supadata_request.assert_not_called()


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
