"""
Ingestion pipeline — load documents, split, embed, and store in ChromaDB.

Usage:
    python create_database.py                        # default: documents/ps1.pdf
    python create_database.py path/to/file.pdf
    python create_database.py documents/*.pdf
    python create_database.py https://en.wikipedia.org/wiki/Retrieval-augmented_generation
"""

import sys
import glob as _glob
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from rag import config
from rag.ingest import load_documents


def ingest(sources: list[str]) -> None:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    all_chunks = []
    for source in sources:
        print(f"  Loading  {source}")
        docs = load_documents(source)
        chunks = splitter.split_documents(docs)
        all_chunks.extend(chunks)
        print(f"           {len(docs)} page(s) → {len(chunks)} chunk(s)")

    if not all_chunks:
        print("Nothing to ingest.")
        return

    print(f"\n  Embedding {len(all_chunks)} total chunk(s)...")
    embeddings = OpenAIEmbeddings(model=config.EMBEDDING_MODEL)
    Chroma.from_documents(
        documents=all_chunks,
        embedding=embeddings,
        persist_directory=config.PERSIST_DIR,
    )
    print(f"  Saved to  '{config.PERSIST_DIR}'\n")


if __name__ == "__main__":
    raw = sys.argv[1:] or ["documents/ps1.pdf"]

    # Expand shell globs that weren't expanded by the shell (e.g. on Windows)
    sources: list[str] = []
    for arg in raw:
        expanded = _glob.glob(arg)
        sources.extend(expanded if expanded else [arg])

    print(f"\nIngesting {len(sources)} source(s):\n")
    ingest(sources)
