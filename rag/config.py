"""
Central configuration — all settings in one place.
Override any value via environment variable or .env file.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# Vector store
PERSIST_DIR: str = os.getenv("PERSIST_DIR", "./chroma_db")

# Models
MODEL: str = os.getenv("MODEL", "grok-4-fast-non-reasoning")
EMBEDDING_MODEL: str = "text-embedding-3-small"

# Chunking
CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", "200"))

# Retrieval
RETRIEVAL_K: int = int(os.getenv("RETRIEVAL_K", "4"))

# API keys
XAI_API_KEY: str = os.getenv("XAI_API_KEY", "")
OPENAI_API_KEY: str = os.getenv("OPENAI_API_KEY", "")
