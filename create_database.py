"""
Ingestion pipeline — load documents, split, embed, and store in ChromaDB.

Usage:
    python create_database.py                        # default: documents/ps1.pdf
    python create_database.py path/to/file.pdf
    python create_database.py documents/*.pdf
    python create_database.py --append notes.md      # add to the existing store
    python create_database.py https://en.wikipedia.org/wiki/Retrieval-augmented_generation
"""

import argparse
import glob
import sys

from rag import config
from rag.chain import open_vectorstore
from rag.embeddings import openai_embeddings
from rag.ingest import build_index, load_documents, split_documents


def main() -> int:
    parser = argparse.ArgumentParser(description="Build the vector store.")
    parser.add_argument("sources", nargs="*", default=["documents/ps1.pdf"])
    parser.add_argument("--append", action="store_true", help="keep existing chunks instead of rebuilding")
    args = parser.parse_args()

    # Expand globs the shell didn't (e.g. on Windows)
    sources = [p for arg in args.sources for p in (glob.glob(arg) or [arg])]

    try:
        embeddings = openai_embeddings()
        chunks = []
        print(f"\nIngesting {len(sources)} source(s):\n")
        for source in sources:
            print(f"  Loading  {source}")
            docs = load_documents(source)
            split = split_documents(docs)
            chunks.extend(split)
            print(f"           {len(docs)} page(s) → {len(split)} chunk(s)")
        if not chunks:
            print("Nothing to ingest.")
            return 1
        print(f"\n  Embedding {len(chunks)} chunk(s)...")
        total = build_index(chunks, open_vectorstore(embeddings), reset=not args.append)
    except (config.ConfigError, FileNotFoundError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    print(f"  Saved to  '{config.PERSIST_DIR}' ({total} chunks in store)\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
