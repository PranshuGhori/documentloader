"""
Streamlit chat UI — streaming answers, source citations, conversation memory.

Run:
    streamlit run app.py
"""

import streamlit as st
from langchain_core.messages import AIMessage, HumanMessage

from rag import chain, config

st.set_page_config(page_title="RAG Q&A", layout="wide")


def show_sources(docs) -> None:
    if not docs:
        return
    with st.expander(f"View {len(docs)} source(s)", expanded=False):
        for s in chain.format_sources(docs, excerpt=300):
            label = f"**Source {s['index']}** — `{s['source']}`"
            if s["page"] is not None:
                label += f"  ·  page {s['page']}"
            st.markdown(label)
            st.caption(s["content"])


# ── Guard ─────────────────────────────────────────────────────────────────────

try:
    stats = chain.vectorstore_stats()
    chain.default_llm()
except config.ConfigError as e:
    st.error(str(e))
    st.stop()

if stats["chunks"] == 0:
    st.error("Vector store is empty. Run `python create_database.py` first.")
    st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("RAG Q&A")
    st.caption("Grok + ChromaDB")
    st.divider()
    col1, col2 = st.columns(2)
    col1.metric("Chunks", stats["chunks"])
    col2.metric("Top-k", stats["retrieval_k"])
    st.caption(f"Model: `{stats['model']}`")
    st.caption(f"Embeddings: `{stats['embedding_model']}`")
    st.divider()
    if st.button("Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.history = []
        st.rerun()

# ── Session state ─────────────────────────────────────────────────────────────

st.session_state.setdefault("messages", [])
st.session_state.setdefault("history", [])

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        show_sources(msg.get("sources"))

# ── Chat input ────────────────────────────────────────────────────────────────

if question := st.chat_input("Ask a question about your documents…"):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        try:
            docs = chain.retrieve(question)
            answer = st.write_stream(chain.stream_answer(question, docs, st.session_state.history))
        except Exception as e:
            st.error(f"Request failed: {e}")
            st.stop()
        show_sources(docs)

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": docs})
    st.session_state.history = chain.trim_history(
        st.session_state.history + [HumanMessage(question), AIMessage(answer)]
    )
