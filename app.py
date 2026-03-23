"""
Streamlit chat UI — streaming answers, source citations, conversation memory.

Run:
    streamlit run app.py
"""

import os
import streamlit as st
from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, AIMessage
from rag import chain, config

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(page_title="RAG Q&A", page_icon="📄", layout="wide")

# ── Guard ─────────────────────────────────────────────────────────────────────

if not os.path.exists(config.PERSIST_DIR):
    st.error("Vector store not found. Run `python create_database.py` first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("📄 RAG Q&A")
    st.caption("Powered by Grok + ChromaDB")
    st.divider()

    try:
        stats = chain.vectorstore_stats()
        col1, col2 = st.columns(2)
        col1.metric("Chunks", stats["chunks"])
        col2.metric("Top-k", stats["retrieval_k"])
        st.caption(f"Model: `{stats['model']}`")
        st.caption(f"Embeddings: `{stats['embedding_model']}`")
    except Exception:
        st.warning("Could not load vector store stats.")

    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────

if "messages" not in st.session_state:
    st.session_state.messages = []
if "history" not in st.session_state:
    st.session_state.history = []

# ── Render chat history ───────────────────────────────────────────────────────

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            _docs = msg["sources"]
            with st.expander(f"View {len(_docs)} source(s)", expanded=False):
                for i, doc in enumerate(_docs, 1):
                    source = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page")
                    label = f"**Source {i}** — `{source}`"
                    if page is not None:
                        label += f"  ·  page {page + 1}"
                    st.markdown(label)
                    st.caption(doc.page_content[:300] + "…")
                    if i < len(_docs):
                        st.divider()

# ── Chat input ────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask a question about your documents…"):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    # Retrieve sources + stream answer
    docs = chain.retrieve(question)

    with st.chat_message("assistant"):
        answer = st.write_stream(
            chain.stream_answer(question, docs, st.session_state.history)
        )

        if docs:
            with st.expander(f"View {len(docs)} source(s)", expanded=False):
                for i, doc in enumerate(docs, 1):
                    source = os.path.basename(doc.metadata.get("source", "unknown"))
                    page = doc.metadata.get("page")
                    label = f"**Source {i}** — `{source}`"
                    if page is not None:
                        label += f"  ·  page {page + 1}"
                    st.markdown(label)
                    st.caption(doc.page_content[:300] + "…")
                    if i < len(docs):
                        st.divider()

    # Persist to session state
    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": docs}
    )
    st.session_state.history.extend([
        HumanMessage(content=question),
        AIMessage(content=answer),
    ])
    # Keep last 10 turns
    if len(st.session_state.history) > 20:
        st.session_state.history = st.session_state.history[-20:]
