"""
Embedding models: OpenAI for real use, a hashing embedder for offline tests and eval.
"""

import hashlib
import math
import re

from langchain_core.embeddings import Embeddings
from langchain_openai import OpenAIEmbeddings

from . import config


def openai_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(model=config.EMBEDDING_MODEL, api_key=config.require("OPENAI_API_KEY"))


class HashingEmbeddings(Embeddings):
    """Deterministic bag-of-words embedder (unigrams + bigrams, hashed, L2-normalised).

    No network, no model weights. A lexical baseline, not a semantic one.
    """

    def __init__(self, dim: int = 1024):
        self.dim = dim

    def _embed(self, text: str) -> list[float]:
        words = re.findall(r"[a-z0-9]+", text.lower())
        vec = [0.0] * self.dim
        for term in words + [f"{a} {b}" for a, b in zip(words, words[1:], strict=False)]:
            h = int.from_bytes(hashlib.md5(term.encode()).digest()[:8], "little")
            vec[h % self.dim] += 1.0 if (h >> 63) else -1.0
        norm = math.sqrt(sum(v * v for v in vec)) or 1.0
        return [v / norm for v in vec]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._embed(t) for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._embed(text)
