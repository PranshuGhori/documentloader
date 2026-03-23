"""
Core RAG logic: retrieval, generation, and streaming.
All three interfaces (CLI, API, Streamlit) import from here.
"""

import os
from typing import Iterator
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_xai import ChatXAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.documents import Document
from langchain_core.messages import BaseMessage
from . import config

SYSTEM_PROMPT = """\
You are a precise, helpful assistant. Answer questions using only the provided context.
When citing information, reference the source label (e.g. "According to Source 2...").
If the answer is not in the context, say you don't know — never make things up.

Context:
{context}"""


# ── Helpers ──────────────────────────────────────────────────────────────────

def _format_docs(docs: list[Document]) -> str:
    parts = []
    for i, doc in enumerate(docs, 1):
        meta = doc.metadata
        source = os.path.basename(meta.get("source", "unknown"))
        page = meta.get("page")
        label = f"Source {i} — {source}" + (f" (page {page + 1})" if page is not None else "")
        parts.append(f"[{label}]\n{doc.page_content}")
    return "\n\n".join(parts)


def _build_prompt(history: list[BaseMessage] | None) -> ChatPromptTemplate:
    messages: list = [("system", SYSTEM_PROMPT)]
    if history:
        messages.extend(history)
    messages.append(("human", "{question}"))
    return ChatPromptTemplate.from_messages(messages)


def _get_retriever():
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vectorstore = Chroma(persist_directory=config.PERSIST_DIR, embedding_function=embeddings)
    return vectorstore.as_retriever(search_kwargs={"k": config.RETRIEVAL_K})


def _get_llm() -> ChatXAI:
    return ChatXAI(model=config.MODEL, xai_api_key=config.XAI_API_KEY)


# ── Public API ────────────────────────────────────────────────────────────────

def retrieve(question: str) -> list[Document]:
    """Retrieve the top-k most relevant document chunks for a question."""
    return _get_retriever().invoke(question)


def answer(
    question: str,
    docs: list[Document],
    history: list[BaseMessage] | None = None,
) -> str:
    """Generate a full answer (blocking) given retrieved docs."""
    context = _format_docs(docs)
    prompt = _build_prompt(history)
    messages = prompt.format_messages(context=context, question=question)
    response = _get_llm().invoke(messages)
    return response.content


def stream_answer(
    question: str,
    docs: list[Document],
    history: list[BaseMessage] | None = None,
) -> Iterator[str]:
    """Stream answer tokens one by one."""
    context = _format_docs(docs)
    prompt = _build_prompt(history)
    messages = prompt.format_messages(context=context, question=question)
    for chunk in _get_llm().stream(messages):
        yield chunk.content


def vectorstore_stats() -> dict:
    """Return basic stats about the persisted vector store."""
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    vs = Chroma(persist_directory=config.PERSIST_DIR, embedding_function=embeddings)
    return {
        "chunks": vs._collection.count(),
        "persist_dir": config.PERSIST_DIR,
        "model": config.MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "retrieval_k": config.RETRIEVAL_K,
    }
