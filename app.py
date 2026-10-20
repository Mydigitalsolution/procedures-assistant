"""
Departmental Procedures Assistant
----------------------------------
Upload PDF procedure documents to build a permanent, growing knowledge base,
then ask questions in words or with a photo (taken with your camera or
attached from your device). Answers are generated ONLY from the indexed
documents — never from the model's general knowledge. A photo is used only to
understand what your question is about.

The knowledge base lives on disk in a "chroma_db" folder next to this file, so
it survives closing and reopening the app. Photos are NOT stored: they are kept
in memory for your browser session only.
"""

import os

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI

from rag import (
    DEFAULT_PHOTO_QUESTION,
    answer,
    embed,
    prepare_image,
    read_pdf_chunks,
)

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")
IMAGE_TYPES = ["jpg", "jpeg", "png", "webp"]

st.set_page_config(
    page_title="Departmental Procedures Assistant",
    page_icon="📘",
    layout="wide",
)


# --------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    st.error(
        "No OPENAI_API_KEY found. Create a `.env` file next to app.py "
        "(see `.env.example`) with your own OpenAI API key, then restart the app."
    )
    st.stop()

client = OpenAI(api_key=api_key)


@st.cache_resource
def get_collection():
    # PersistentClient writes to disk under CHROMA_DIR, so the knowledge base
    # survives app restarts and is shared by everyone using this running app.
    persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return persistent_client.get_or_create_collection("procedures")


collection = get_collection()

if "history" not in st.session_state:
    # entries: {"role", "content", "image" (bytes|None), "sources", "photo_summary"}
    st.session_state.history = []

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

if "camera_key" not in st.session_state:
    st.session_state.camera_key = 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def get_indexed_sources():
    """Return {filename: chunk_count} for everything currently in the knowledge base."""
    if collection.count() == 0:
        return {}
    got = collection.get(include=["metadatas"])
    counts = {}
    for meta in got["metadatas"]:
        counts[meta["source"]] = counts.get(meta["source"], 0) + 1
    return counts


def add_documents(uploaded_files):
    """Chunk, embed, and upsert the uploaded PDFs into the persistent knowledge base.

    IDs are deterministic (filename + page + chunk index), so re-uploading a file
    with the same name refreshes its chunks instead of duplicating them.
    """
    all_chunks = []
    for uploaded_file in uploaded_files:
        all_chunks.extend(read_pdf_chunks(uploaded_file.getvalue(), uploaded_file.name))

    if not all_chunks:
        st.error(
            "No extractable text was found in the uploaded PDF(s). "
            "They may be scanned images without a text layer."
        )
        return False

    total = len(all_chunks)
    progress = st.progress(0.0, text="Embedding chunks...")
    for idx, chunk in enumerate(all_chunks):
        chunk_id = f"{chunk['source']}::p{chunk['page']}::c{chunk['chunk_index']}"
        collection.upsert(
            ids=[chunk_id],
            embeddings=[embed(client, chunk["text"])],
            documents=[chunk["text"]],
            metadatas=[{"source": chunk["source"], "page": chunk["page"]}],
        )
        progress.progress((idx + 1) / total, text=f"Embedding chunk {idx + 1}/{total}...")
    progress.empty()
    return True


def clear_knowledge_base():
    existing = collection.get()
    if existing["ids"]:
        collection.delete(ids=existing["ids"])


def render_extras(entry):
    """Photo summary + sources under an assistant answer."""
    if entry.get("photo_summary"):
        with st.expander("🔍 What I saw in your photo"):
            st.write(entry["photo_summary"])
    sources = entry.get("sources")
    if sources:
        with st.expander(f"📄 Sources ({len(sources)})"):
            for meta in sources:
                st.caption(f"{meta['source']} — Page {meta['page']}")


# --------------------------------------------------------------------------
# Sidebar — knowledge base management
# --------------------------------------------------------------------------

with st.sidebar:
    st.markdown("### 📚 Knowledge base")

    source_counts = get_indexed_sources()
    col1, col2 = st.columns(2)
    col1.metric("Documents", len(source_counts))
    col2.metric("Chunks", collection.count())

    st.divider()
    st.markdown("### ➕ Add documents")
    st.caption("Upload PDFs to add them to the knowledge base. Re-uploading a file with the same name refreshes it.")

    uploaded_files = st.file_uploader(
        "PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        key=f"uploader_{st.session_state.uploader_key}",
        label_visibility="collapsed",
    )

    if st.button("Add to knowledge base", type="primary", disabled=not uploaded_files, width="stretch"):
        with st.spinner("Indexing..."):
            success = add_documents(uploaded_files)
        if success:
            st.session_state.uploader_key += 1
            st.toast(f"Added {len(uploaded_files)} document(s) to the knowledge base.", icon="✅")
            st.rerun()

    if st.session_state.history:
        st.divider()
        if st.button("🗑️ Clear chat", width="stretch"):
            st.session_state.history = []
            st.rerun()

    if source_counts:
        st.divider()
        with st.expander("⚠️ Danger zone"):
            confirm = st.checkbox("I understand this permanently deletes ALL indexed documents")
            if st.button("Clear entire knowledge base", disabled=not confirm, width="stretch"):
                clear_knowledge_base()
                st.session_state.history = []
                st.rerun()


# --------------------------------------------------------------------------
# Main — ask questions (text and/or photo)
# --------------------------------------------------------------------------

st.title("📘 Departmental Procedures Assistant")
st.caption(
    "Ask in words, or add a photo with your question. "
    "Answers come only from the documents in your knowledge base."
)

if collection.count() == 0:
    with st.container(border=True):
        st.markdown("#### 👋 Get started")
        st.markdown(
            "1. Open the sidebar and upload one or more PDF procedure documents.\n"
            "2. Click **Add to knowledge base**.\n"
            "3. Come back here and ask questions — type them, or take a photo of what you're asking about.\n\n"
            "Once added, your documents stay indexed even after you close and reopen the app."
        )
    st.stop()

camera_widget_key = f"camera_{st.session_state.camera_key}"
# The invisible suffix changes after each camera photo is sent, which makes the
# expander start collapsed again (so the camera isn't left running).
with st.expander("📷 Take a photo with your camera" + "\u200b" * (st.session_state.camera_key % 2)):
    camera_shot = st.camera_input(
        "Take a photo",
        key=camera_widget_key,
        label_visibility="collapsed",
        help="Your browser will ask for permission to use the camera.",
    )
    st.caption(
        "Take the photo, then type your question in the box below and send it. "
        "On a phone you can also use the 📎 button in the message box to take or choose a photo."
    )

if camera_shot is not None:
    st.info("📷 Photo ready — type your question below and press send (you can also send it with no text).")

for entry in st.session_state.history:
    with st.chat_message(entry["role"]):
        if entry.get("image"):
            st.image(entry["image"], width=260)
        st.write(entry["content"])
        if entry["role"] == "assistant":
            render_extras(entry)

prompt = st.chat_input(
    "Ask a question about your procedures...",
    accept_file=True,
    file_type=IMAGE_TYPES,
)

if prompt:
    # Works whether Streamlit hands back an object (text + files) or a plain string.
    text = (prompt if isinstance(prompt, str) else (prompt.text or "")).strip()
    attached = None if isinstance(prompt, str) else (prompt.files[0] if prompt.files else None)

    # An attached image wins; otherwise use the camera photo, if one was taken.
    used_camera = attached is None and camera_shot is not None
    source_file = attached if attached is not None else camera_shot

    image = None
    if source_file is not None:
        try:
            image = prepare_image(source_file.getvalue())
        except ValueError as e:
            st.error(str(e))
            st.stop()

    if not text and image is None:
        st.stop()

    shown_text = text or DEFAULT_PHOTO_QUESTION

    with st.chat_message("user"):
        if image:
            st.image(image, width=260)
        st.write(shown_text)

    with st.chat_message("assistant"):
        spinner_text = (
            "Looking at your photo and searching the procedures..."
            if image
            else "Searching procedure documents..."
        )
        with st.spinner(spinner_text):
            try:
                result = answer(client, collection, text, image)
            except Exception as e:
                st.error(
                    "Sorry — something went wrong while answering "
                    f"({type(e).__name__}). Please try again in a moment."
                )
                st.stop()

    st.session_state.history.append(
        {"role": "user", "content": shown_text, "image": image}
    )
    st.session_state.history.append({
        "role": "assistant",
        "content": result["answer"],
        "sources": result["sources"],
        "photo_summary": result["photo_summary"],
    })

    if used_camera:
        st.session_state.camera_key += 1  # fresh camera for the next question
    st.rerun()
