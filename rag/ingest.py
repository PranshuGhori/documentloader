"""
Load, split, and index documents.
Supports: PDF, TXT, Markdown, web URLs.
"""

from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from . import config


def load_documents(source: str) -> list[Document]:
    """Load documents from a PDF path, plain text file, or http(s) URL."""
    source = source.strip()

    if source.startswith(("http://", "https://")):
        from langchain_community.document_loaders import WebBaseLoader

        return WebBaseLoader(source).load()

    path = Path(source)
    if not path.is_file():
        raise FileNotFoundError(f"File not found: {source}")

    ext = path.suffix.lower()
    if ext == ".pdf":
        return PyPDFLoader(str(path)).load()
    if ext in (".txt", ".md"):
        return TextLoader(str(path), encoding="utf-8").load()

    raise ValueError(f"Unsupported file type '{ext}'. Supported: .pdf, .txt, .md, or a URL.")


def split_documents(
    docs: list[Document],
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
) -> list[Document]:
    """Split into chunks and give each a stable id: '<source>#p<page>:c<n>'."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size or config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP if chunk_overlap is None else chunk_overlap,
    )
    chunks = []
    for doc in docs:
        for n, chunk in enumerate(splitter.split_documents([doc])):
            page = chunk.metadata.get("page")
            chunk.metadata["chunk"] = n
            chunk.id = f"{chunk.metadata.get('source', 'unknown')}#p{page if page is not None else '-'}:c{n}"
            chunks.append(chunk)
    return chunks


def build_index(chunks: list[Document], vectorstore: Chroma, reset: bool = True) -> int:
    """Write chunks to the vector store. Stable ids make re-ingesting idempotent."""
    if reset:
        vectorstore.reset_collection()
    if chunks:
        vectorstore.add_documents(chunks, ids=[c.id for c in chunks])
    return vectorstore._collection.count()
