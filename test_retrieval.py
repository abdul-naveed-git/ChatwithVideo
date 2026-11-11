#!/usr/bin/env python
"""
test_retrieval.py – LangChain test script for ingestion and retrieval.

Usage:
    python test_retrieval.py <YOUTUBE_URL> [--query "question"]
"""

from __future__ import annotations

import argparse
import sys
from ingestion.transcripts import load_youtube_transcript_documents
from ingestion.chunking import split_transcript_documents
from retrieval.store import add_documents_to_store, get_retriever
from generation.answer import ask_youtube


def main() -> None:
    parser = argparse.ArgumentParser(description="Test LangChain YouTube RAG")
    parser.add_argument("url", help="YouTube video URL or playlist URL")
    parser.add_argument("--query", "-q", default="What is this video about?", help="Question")
    args = parser.parse_args()

    print(f"\n1. Loading transcript documents for: {args.url}")
    docs = load_youtube_transcript_documents(args.url)
    print(f"   Loaded {len(docs)} transcript segments.")

    if not docs:
        print("   No captions found. Exiting.")
        sys.exit(1)

    print("\n2. Splitting with LangChain RecursiveCharacterTextSplitter...")
    chunks = split_transcript_documents(docs)
    print(f"   Created {len(chunks)} chunks.")

    print("\n3. Storing chunks in Chroma vectorstore (BAAI/bge-m3 embeddings)...")
    add_documents_to_store(chunks)
    print("   Stored in Chroma successfully.")

    print(f"\n4. Retrieving relevant chunks for query: '{args.query}'")
    retriever = get_retriever(k=3)
    results = retriever.invoke(args.query)

    for i, r in enumerate(results, 1):
        print(f"\n--- Result {i} [{r.metadata.get('title')} @ {r.metadata.get('start', 0):.0f}s] ---")
        print(r.page_content[:300])

    print("\n✓ LangChain ingestion and retrieval completed successfully.")


if __name__ == "__main__":
    main()
