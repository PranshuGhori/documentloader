"""
Central configuration — all settings in one place.
Override any value via environment variable or .env file.
"""

import os

from dotenv import load_dotenv

load_dotenv()


class ConfigError(RuntimeError):
    """Invalid or missing configuration."""


def _int(name: str, default: int, minimum: int) -> int:
    raw = os.getenv(name, str(default))
    try:
        value = int(raw)
    except ValueError:
        raise ConfigError(f"{name} must be an integer, got {raw!r}") from None
    if value < minimum:
        raise ConfigError(f"{name} must be >= {minimum}, got {value}")
    return value


def require(name: str) -> str:
    """Return a required secret from the environment or raise a clear error."""
    value = os.getenv(name, "").strip()
    if not value:
        raise ConfigError(f"{name} is not set. Add it to .env (see .env.example).")
    return value


# Vector store
PERSIST_DIR: str = os.getenv("PERSIST_DIR", "./chroma_db")
COLLECTION: str = os.getenv("COLLECTION", "documents")

# Models
MODEL: str = os.getenv("MODEL", "grok-4-fast-non-reasoning")
EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# Chunking
CHUNK_SIZE: int = _int("CHUNK_SIZE", 1000, minimum=1)
CHUNK_OVERLAP: int = _int("CHUNK_OVERLAP", 200, minimum=0)
if CHUNK_OVERLAP >= CHUNK_SIZE:
    raise ConfigError(f"CHUNK_OVERLAP ({CHUNK_OVERLAP}) must be smaller than CHUNK_SIZE ({CHUNK_SIZE})")

# Retrieval and memory
RETRIEVAL_K: int = _int("RETRIEVAL_K", 4, minimum=1)
HISTORY_TURNS: int = _int("HISTORY_TURNS", 5, minimum=0)
