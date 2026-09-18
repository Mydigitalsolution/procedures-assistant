"""
Departmental Procedures Assistant
----------------------------------
Upload PDF procedure documents in the browser to build a permanent,
growing knowledge base. Answers are generated ONLY from documents that
have been added to it — never from the model's general knowledge.

The knowledge base is stored on disk in a "chroma_db" folder next to this
file, so it survives closing and reopening the app: you only need to
upload a document once. Uploading again adds to (or, for a file with the
same name, refreshes) the knowledge base rather than replacing it.
"""

import io
import os

import chromadb
import streamlit as st
from dotenv import load_dotenv
from openai import OpenAI
from pypdf import PdfReader

load_dotenv()

EMBED_MODEL = "text-embedding-3-small"
CHAT_MODEL = "gpt-4.1-mini"
CHUNK_WORDS = 300
TOP_K = 5

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHROMA_DIR = os.path.join(BASE_DIR, "chroma_db")

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
    # PersistentClient writes to disk under CHROMA_DIR, so the knowledge
    # base survives app restarts and is shared by everyone who uses this
    # running app (not per-browser-session like before).
    persistent_client = chromadb.PersistentClient(path=CHROMA_DIR)
    return persistent_client.get_or_create_collection("procedures")


collection = get_collection()

if "history" not in st.session_state:
    st.session_state.history = []  # list of {"role": "user"/"assistant", "content": ..., "sources": [...]}

if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def embed(text: str):
    response = client.embeddings.create(model=EMBED_MODEL, input=text)
    return response.data[0].embedding


def read_pdf_chunks(file_bytes: bytes, filename: str):
    """Extract text from a PDF (in memory) and split into ~300-word chunks per page."""
    reader = PdfReader(io.BytesIO(file_bytes))
    chunks = []

    for page_num, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        words = text.split()

        for local_idx, i in enumerate(range(0, len(words), CHUNK_WORDS)):
            chunk = " ".join(words[i:i + CHUNK_WORDS])
            if chunk.strip():
                chunks.append({
                    "text": chunk,
                    "page": page_num,
                    "source": filename,
                    "chunk_index": local_idx,
                })

    return chunks


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
    """Chunk, embed, and upsert the uploaded files into the persistent knowledge base.

    Uses deterministic IDs (filename + page + chunk index), so re-uploading a
    file with the same name refreshes its existing chunks instead of
    duplicating them; a brand new filename is simply added alongside what's
    already indexed.
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
            embeddings=[embed(chunk["text"])],
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


def answer_question(question: str):
    question_embedding = embed(question)

    results = collection.query(query_embeddings=[question_embedding], n_results=TOP_K)
    docs = results["documents"][0]
    metas = results["metadatas"][0]

    if not docs:
        return "I could not find this in the uploaded departmental procedures.", []

    context = "\n\n".join(
        f"Source: {meta['source']}, Page: {meta['page']}\n{doc}"
        for doc, meta in zip(docs, metas)
    )

    prompt = f"""
You are a departmental procedures assistant.

Rules:
1. Answer only using the provided context.
2. Do not use outside knowledge.
3. If the answer is not found, say:
   "I could not find this in the uploaded departmental procedures."
4. Always cite the source document and page number.
5. Be clear, practical, and step-by-step when explaining procedures.

Context:
{context}

Question:
{question}
"""

    response = client.responses.create(model=CHAT_MODEL, input=prompt)
    return response.output_text, metas


def render_sources(sources):
    if not sources:
        return
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
# Main — ask questions
# --------------------------------------------------------------------------

st.title("📘 Departmental Procedures Assistant")
st.caption("Answers are grounded only in the documents indexed in your knowledge base.")

if collection.count() == 0:
    with st.container(border=True):
        st.markdown("#### 👋 Get started")
        st.markdown(
            "1. Open the sidebar and upload one or more PDF procedure documents.\n"
            "2. Click **Add to knowledge base**.\n"
            "3. Come back here and start asking questions.\n\n"
            "Once added, your documents stay indexed even after you close and reopen the app."
        )
    st.stop()

for msg in st.session_state.history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])
        if msg["role"] == "assistant":
            render_sources(msg.get("sources"))

question = st.chat_input("Ask a question about your procedures...")

if question:
    st.session_state.history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching procedure documents..."):
            answer, sources = answer_question(question)
        st.write(answer)
        render_sources(sources)

    st.session_state.history.append({"role": "assistant", "content": answer, "sources": sources})
