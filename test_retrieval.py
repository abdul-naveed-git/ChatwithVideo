
from __future__ import annotations
from ingestion.transcripts import load_youtube_transcript_documents
from retrieval.store import add_documents_to_store, get_retriever
from generation.answer import create_rag_chain


def main() -> None:
    url="https://youtu.be/J7j5tCB_y4w?si=ylVmQH0TAf9iOL-m"
    print(f"\n1. Loading transcript documents for: {url}")
    docs = load_youtube_transcript_documents(url)
    print(f"   Loaded {len(docs)} transcript segments.")

    if not docs:
        print("   No captions found. Exiting.")
        return

    print("\n2. Assembling captions and storing LangChain parent/child chunks in Chroma...")
    add_documents_to_store(docs)
    print("   Stored in Chroma successfully.")

    query="Langchain అంటే ఏమిటి??"
    print(f"\n3. Retrieving relevant parent passages for query: '{query}'")
    video_ids = sorted({doc.metadata["video_id"] for doc in docs})
    retriever = get_retriever(k=5,video_ids=video_ids)
    results = retriever.invoke(query)

    for i, r in enumerate(results, 1):
        start = r.metadata.get("start")
        parent_id = r.metadata.get("parent_id", "unknown parent key")
        stamp = f"{float(start):.2f}s" if start is not None else "timestamp unavailable"
        print(f"\n--- Result {i} [{r.metadata.get('title')} @ {stamp}; parent={parent_id}] ---")
        print(r.page_content)

    print("\n✓ LangChain ingestion and retrieval completed successfully.")
    print("\n4. Generating a grounded Gemini answer from the retrieved parent passages...")
    rag_chain = create_rag_chain(retriever)
    answer = rag_chain.invoke({
        "question": query,
        "response_language": "English",
    })
    print("\n--- Gemini answer ---")
    print(answer)
    print("\n✓ LangChain ingestion, retrieval, and Gemini generation completed successfully.")

if __name__ == "__main__":
    main()
