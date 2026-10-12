"""
Core RAG logic: retrieval, generation, and streaming.
All three interfaces (CLI, API, Streamlit) import from here.

The vector store and LLM default to Chroma + OpenAI embeddings and Grok, and can
be passed in explicitly (tests use a hashing embedder and a fake LLM).
"""

import os
from collections.abc import Iterator
from functools import lru_cache

import chromadb
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import BaseMessage
from langchain_core.prompts import ChatPromptTemplate

from . import config
from .embeddings import openai_embeddings

SYSTEM_PROMPT = """\
You are a precise, helpful assistant. Answer questions using only the provided context.
When citing information, reference the source label (e.g. "According to Source 2...").
If the answer is not in the context, say you don't know — never make things up.

Context:
{context}"""


class EmptyStoreError(RuntimeError):
    """The vector store has no documents."""


# ── Components ───────────────────────────────────────────────────────────────


def open_vectorstore(embeddings: Embeddings, persist_dir: str | None = None) -> Chroma:
    client = chromadb.PersistentClient(
        path=persist_dir or config.PERSIST_DIR,
        settings=chromadb.Settings(anonymized_telemetry=False),
    )
    return Chroma(client=client, collection_name=config.COLLECTION, embedding_function=embeddings)


@lru_cache(maxsize=1)
def default_vectorstore() -> Chroma:
    return open_vectorstore(openai_embeddings())


@lru_cache(maxsize=1)
def default_llm() -> BaseChatModel:
    from langchain_xai import ChatXAI

    return ChatXAI(model=config.MODEL, xai_api_key=config.require("XAI_API_KEY"))


# ── Helpers ──────────────────────────────────────────────────────────────────


def source_label(doc: Document) -> str:
    """'ps1.pdf, page 3' — pages are stored 0-based, shown 1-based."""
    name = os.path.basename(doc.metadata.get("source", "unknown"))
    page = doc.metadata.get("page")
    return f"{name}, page {page + 1}" if isinstance(page, int) else name


def format_context(docs: list[Document]) -> str:
    if not docs:
        return "(no relevant context found)"
    return "\n\n".join(f"[Source {i}: {source_label(d)}]\n{d.page_content}" for i, d in enumerate(docs, 1))


def format_sources(docs: list[Document], excerpt: int = 400) -> list[dict]:
    """Citation records, numbered to match the 'Source N' labels the model sees."""
    out = []
    for i, d in enumerate(docs, 1):
        page = d.metadata.get("page")
        text = d.page_content
        out.append(
            {
                "index": i,
                "source": os.path.basename(d.metadata.get("source", "unknown")),
                "page": page + 1 if isinstance(page, int) else None,
                "content": text if len(text) <= excerpt else text[:excerpt].rstrip() + "…",
            }
        )
    return out


def trim_history(history: list[BaseMessage], turns: int | None = None) -> list[BaseMessage]:
    """Keep the last `turns` question/answer pairs."""
    turns = config.HISTORY_TURNS if turns is None else turns
    return history[-2 * turns :] if turns else []


def _messages(question: str, docs: list[Document], history: list[BaseMessage] | None) -> list[BaseMessage]:
    prompt = ChatPromptTemplate.from_messages(
        [("system", SYSTEM_PROMPT), *trim_history(history or []), ("human", "{question}")]
    )
    return prompt.format_messages(context=format_context(docs), question=question)


# ── Public API ────────────────────────────────────────────────────────────────


def retrieve(question: str, vectorstore: Chroma | None = None, k: int | None = None) -> list[Document]:
    """Retrieve the top-k most relevant document chunks for a question."""
    vs = default_vectorstore() if vectorstore is None else vectorstore
    if vs._collection.count() == 0:
        raise EmptyStoreError("The vector store is empty. Run 'python create_database.py' first.")
    return vs.similarity_search(question, k=k or config.RETRIEVAL_K)


def answer(
    question: str,
    docs: list[Document],
    history: list[BaseMessage] | None = None,
    llm: BaseChatModel | None = None,
) -> str:
    """Generate a full answer (blocking) given retrieved docs."""
    return (default_llm() if llm is None else llm).invoke(_messages(question, docs, history)).content


def stream_answer(
    question: str,
    docs: list[Document],
    history: list[BaseMessage] | None = None,
    llm: BaseChatModel | None = None,
) -> Iterator[str]:
    """Stream answer tokens as they arrive."""
    for chunk in (default_llm() if llm is None else llm).stream(_messages(question, docs, history)):
        if chunk.content:
            yield chunk.content


def vectorstore_stats(vectorstore: Chroma | None = None) -> dict:
    """Return basic stats about the persisted vector store."""
    vs = default_vectorstore() if vectorstore is None else vectorstore
    return {
        "chunks": vs._collection.count(),
        "persist_dir": config.PERSIST_DIR,
        "model": config.MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "retrieval_k": config.RETRIEVAL_K,
    }
