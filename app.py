"""Streamlit frontend for the YouTube multilingual RAG service."""
from __future__ import annotations
import os
import uuid
from typing import Any
import streamlit as st
import streamlit.components.v1 as components
from config import RESPONSE_LANGUAGES
from generation.answer import ask_youtube
from ingestion.transcripts import extract_video_id, load_youtube_transcript_documents
from retrieval.store import add_documents_to_store, video_is_indexed

st.set_page_config(page_title="Lecture RAG", page_icon="▶", layout="wide", initial_sidebar_state="expanded")
# Streamlit secrets stay server-side. The loader reads the same value from the
# environment, which also keeps the FastAPI deployment configuration simple.
if not os.environ.get("SUPADATA_API_KEY"):
    try:
        os.environ["SUPADATA_API_KEY"] = st.secrets["SUPADATA_API_KEY"]
    except Exception:
        pass
st.markdown("""<style>
[data-testid="stAppViewContainer"]{background:#111315;color:#e8e8e8}[data-testid="stSidebar"]{background:#1a1c1f;border-right:1px solid #303237}.block-container{max-width:1040px;padding-top:1.4rem}[data-testid="stChatMessage"]{background:#1b1e22;border:1px solid #30343a;border-radius:14px;padding:.2rem .65rem}[data-testid="stChatMessage"] p{line-height:1.6}.instruction{color:#b8bcc4;margin:0 0 1rem}.source-label{color:#8ab4f8;font-size:.85rem;margin-top:.5rem}div[data-testid="stButton"]>button[kind="primary"]{background:#b3261e;border-color:#d74239}div[data-testid="stButton"]>button{border-radius:9px}
</style>""", unsafe_allow_html=True)
DEFAULTS: dict[str, Any] = {"messages": [], "lectures": {}, "selected_video_ids": set(), "active_video_id": None, "player_start": 0, "player_nonce": 0, "response_language": "English"}
for key, value in DEFAULTS.items():
    if key not in st.session_state: st.session_state[key] = value.copy() if isinstance(value, (dict, list, set)) else value

def select_player(video_id: str, start: float = 0) -> None:
    st.session_state.active_video_id = video_id
    st.session_state.player_start = max(0, int(start))
    st.session_state.player_nonce += 1  # reloads iframe for repeated same-video citation seeks

def timestamp_label(start: Any) -> str:
    if start is None: return "timestamp unavailable"
    seconds = int(float(start)); return f"{seconds // 60}:{seconds % 60:02d}"

def render_sources(sources: list[dict[str, Any]], message_id: str) -> None:
    if not sources: return
    st.markdown("<p class='source-label'>Sources</p>", unsafe_allow_html=True)
    columns = st.columns(min(3, len(sources)))
    for index, source in enumerate(sources):
        video_id, start = source.get("video_id"), source.get("start")
        if video_id and start is not None:
            with columns[index % len(columns)]:
                st.button(f"▶ {timestamp_label(start)} · {source.get('title') or 'Video'}", key=f"source-{message_id}-{index}-{video_id}-{int(float(start))}", help="Open this lecture at the cited time", on_click=select_player, args=(video_id, float(start)), use_container_width=True)

def render_message(message: dict[str, Any]) -> None:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message["role"] == "assistant":
            render_sources(message.get("sources", []), message["id"])
            debug = message.get("debug_documents", [])
            with st.expander(f"Retrieved passages ({len(debug)})", expanded=False):
                for item in debug:
                    st.caption(f"{item['title']} · {timestamp_label(item['start'])} · {item['parent_id']}")
                    st.text(item["text"])

def remember_lectures(documents: list[Any]) -> None:
    for doc in documents:
        meta, video_id = doc.metadata, doc.metadata["video_id"]
        st.session_state.lectures[video_id] = {"title": meta.get("title") or f"Video {video_id}", "url": meta.get("url", "")}
        st.session_state.selected_video_ids.add(video_id)

with st.sidebar:
    st.title("Lecture RAG")
    url_input = st.text_input("YouTube URL", placeholder="https://youtu.be/…", key="youtube_url")
    add_col, refresh_col = st.columns(2)
    add_clicked = add_col.button("Add lecture", use_container_width=True, type="primary")
    refresh_clicked = refresh_col.button("Re-ingest", use_container_width=True)
    if add_clicked or refresh_clicked:
        video_id = extract_video_id(url_input.strip()) if url_input else None
        if not video_id:
            st.error("Enter a valid YouTube video link.")
        else:
            with st.spinner("Fetching and validating captions…"):
                try:
                    captions = load_youtube_transcript_documents(url_input)
                    if not captions:
                        st.error("No captions are available for this lecture.")
                    elif not refresh_clicked and (video_id in st.session_state.lectures or video_is_indexed(video_id)):
                        remember_lectures(captions); select_player(video_id)
                        st.info("This lecture is already indexed; reused it without creating duplicates.")
                    else:
                        add_documents_to_store(captions)  # replaces this video only, after caption validation
                        remember_lectures(captions); select_player(video_id)
                        st.success(f"{captions[0].metadata.get('title', video_id)} {'refreshed' if refresh_clicked else 'added'} ({len(captions)} captions).")
                except Exception as exc:
                    st.error(f"Could not ingest this lecture: {exc}")
    st.divider(); st.subheader("Your lectures")
    if not st.session_state.lectures: st.caption("Add a captioned YouTube lecture to begin.")
    for video_id, lecture in st.session_state.lectures.items():
        row_check, row_title = st.columns([1, 5])
        checked = row_check.checkbox("Search", value=video_id in st.session_state.selected_video_ids, key=f"select-{video_id}", label_visibility="collapsed")
        if checked: st.session_state.selected_video_ids.add(video_id)
        else: st.session_state.selected_video_ids.discard(video_id)
        row_title.button(("▶ " if video_id == st.session_state.active_video_id else "") + lecture["title"], key=f"lecture-{video_id}", type="primary" if video_id == st.session_state.active_video_id else "secondary", on_click=select_player, args=(video_id, 0), use_container_width=True)
    st.divider(); st.selectbox("Answer in", RESPONSE_LANGUAGES, key="response_language")
    if st.button("Clear chat", use_container_width=True):
        st.session_state.messages = []; st.rerun()

st.markdown("<p class='instruction'>Ask questions across the lectures you’ve selected.</p>", unsafe_allow_html=True)
active_video = st.session_state.active_video_id
if active_video:
    start = st.session_state.player_start
    embed_url = f"https://www.youtube.com/embed/{active_video}?start={start}&autoplay=0&rel=0&v={st.session_state.player_nonce}"
    components.html(f'<iframe title="YouTube lecture" width="100%" height="100%" src="{embed_url}" frameborder="0" allowfullscreen></iframe>', height=500)
    st.markdown(f"[Open this timestamp on YouTube](https://youtu.be/{active_video}?t={start})")
else: st.info("Choose a lecture to open the player.")
for message in st.session_state.messages: render_message(message)

question = st.chat_input("Ask a question about the selected lectures…")
if question:
    selected = sorted(st.session_state.selected_video_ids)
    if not selected: st.warning("Select at least one lecture in the sidebar before asking.")
    else:
        user_message = {"id": str(uuid.uuid4()), "role": "user", "content": question}
        st.session_state.messages.append(user_message); render_message(user_message)
        with st.spinner("Searching selected lectures and generating an answer…"):
            try:
                result = ask_youtube(question=question, video_ids=selected, response_language=st.session_state.response_language, k=5)
                sources = result.get("sources", [])
                for index, source in enumerate(sources): source["source_id"] = f"{source.get('video_id')}-{source.get('start')}-{index}"
                debug_documents = [{"title": d.metadata.get("title", "Video"), "start": d.metadata.get("start"), "parent_id": d.metadata.get("parent_id", "unknown parent"), "text": d.page_content} for d in result.get("documents", [])]
                assistant_message = {"id": str(uuid.uuid4()), "role": "assistant", "content": result["answer"], "sources": sources, "debug_documents": debug_documents}
                st.session_state.messages.append(assistant_message); render_message(assistant_message)
            except Exception as exc: st.error(f"Could not generate an answer: {exc}")
