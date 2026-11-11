"""
Streamlit UI powered by LangChain and ChromaDB.
"""

from __future__ import annotations

import logging
from typing import Any

import streamlit as st

from config import RESPONSE_LANGUAGES
from generation.answer import ask_youtube
from ingestion.chunking import split_transcript_documents
from ingestion.transcripts import (
    extract_video_id,
    list_captions,
    load_youtube_transcript_documents,
)
from retrieval.store import add_documents_to_store, delete_by_video_id, get_vectorstore

logging.basicConfig(level=logging.INFO)

st.set_page_config(
    page_title="YouTube RAG (LangChain + Chroma)",
    page_icon="🎬",
    layout="wide",
)

# ── Session State ──────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "current_video_id" not in st.session_state:
    st.session_state.current_video_id = None
if "ingested_videos" not in st.session_state:
    st.session_state.ingested_videos = {}


# ── Sidebar ───────────────────────────────────
with st.sidebar:
    st.header("🎬 YouTube RAG")
    st.caption("Powered by LangChain LCEL & ChromaDB")

    st.subheader("Add Video")
    url_input = st.text_input("YouTube URL", placeholder="https://youtu.be/...")

    col1, col2 = st.columns(2)
    with col1:
        add_btn = st.button("📥 Ingest", use_container_width=True)
    with col2:
        reingest_btn = st.button("🔄 Re-ingest", use_container_width=True)

    if (add_btn or reingest_btn) and url_input:
        vid = extract_video_id(url_input)
        with st.spinner("Ingesting transcript via LangChain..."):
            try:
                # 1. Load LangChain Documents
                raw_docs = load_youtube_transcript_documents(url_input)
                if not raw_docs:
                    st.error("No captions available for this video.")
                else:
                    # 2. Split with RecursiveCharacterTextSplitter
                    chunks = split_transcript_documents(raw_docs)

                    if reingest_btn and vid:
                        delete_by_video_id(vid)

                    # 3. Store in Chroma
                    add_documents_to_store(chunks)

                    title = raw_docs[0].metadata.get("title", vid or "Video")
                    st.session_state.ingested_videos[vid or url_input] = title
                    st.session_state.current_video_id = vid

                    st.success(f"✅ Ingested '{title}' ({len(chunks)} chunks)")
            except Exception as e:
                st.error(f"Error during ingestion: {e}")

    st.divider()

    st.subheader("Ingested Videos")
    if not st.session_state.ingested_videos:
        st.info("No videos ingested yet.")
    else:
        for vid, title in st.session_state.ingested_videos.items():
            st.markdown(f"📹 **{title}** (`{vid}`)")

    st.divider()

    st.subheader("Response Language")
    resp_lang = st.selectbox(
        "Answer in:",
        RESPONSE_LANGUAGES,
        index=0,
        key="resp_lang_select",
    )

    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Area ──────────────────────────────────
st.title("🎬 YouTube RAG with LangChain & ChromaDB")

# Embedded Player
if st.session_state.current_video_id:
    embed_url = f"https://www.youtube.com/embed/{st.session_state.current_video_id}?autoplay=0"
    st.markdown(
        f'<iframe width="100%" height="380" src="{embed_url}" '
        f'frameborder="0" allow="accelerometer; autoplay; clipboard-write; '
        f'encrypted-media; gyroscope; picture-in-picture" allowfullscreen></iframe>',
        unsafe_allow_html=True,
    )

# Chat messages display
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            st.markdown("**📎 Cited Sources:**")
            for s in msg["sources"]:
                mins = int(s["start"] // 60)
                secs = int(s["start"] % 60)
                st.markdown(f"- [{s['title']} @ {mins}:{secs:02d}]({s['url']})")

# Chat input
user_query = st.chat_input("Ask a question in any language about the video(s)...")
if user_query:
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    with st.chat_message("assistant"):
        with st.spinner("LangChain RAG chain thinking..."):
            try:
                res = ask_youtube(
                    question=user_query,
                    response_language=resp_lang,
                )
                answer_text = res["answer"]
                sources = res["sources"]

                st.markdown(answer_text)

                if sources:
                    st.markdown("**📎 Cited Sources:**")
                    for s in sources:
                        mins = int(s["start"] // 60)
                        secs = int(s["start"] % 60)
                        st.markdown(f"- [{s['title']} @ {mins}:{secs:02d}]({s['url']})")

                with st.expander(f"📄 Retrieved Chunks ({len(res.get('documents', []))})", expanded=False):
                    for i, d in enumerate(res.get("documents", []), 1):
                        st.markdown(f"**[{i}] {d.metadata.get('title')} @ {d.metadata.get('start', 0):.0f}s**")
                        st.text(d.page_content)
                        st.divider()

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer_text,
                    "sources": sources,
                })
            except Exception as exc:
                st.error(f"Error generating answer: {exc}")
