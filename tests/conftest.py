import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, AIMessageChunk

from rag.chain import open_vectorstore
from rag.embeddings import HashingEmbeddings
from rag.ingest import build_index, split_documents

PAGES = [
    "The mitochondria is the powerhouse of the cell and produces ATP.",
    "Photosynthesis in chloroplasts converts sunlight into chemical energy.",
    "The French Revolution began in 1789 with the storming of the Bastille.",
]


class FakeLLM:
    """Records the prompt it was given and replies with a fixed answer."""

    def __init__(self, reply="The answer, per Source 1.", fail=False):
        self.reply, self.fail, self.calls = reply, fail, []

    def invoke(self, messages):
        self.calls.append(messages)
        if self.fail:
            raise RuntimeError("upstream down")
        return AIMessage(self.reply)

    def stream(self, messages):
        self.calls.append(messages)
        for word in self.reply.split(" "):
            yield AIMessageChunk(word + " ")
        if self.fail:
            raise RuntimeError("upstream down")


@pytest.fixture
def docs():
    return [Document(t, metadata={"source": "docs/bio.pdf", "page": i}) for i, t in enumerate(PAGES)]


@pytest.fixture
def empty_store(tmp_path):
    return open_vectorstore(HashingEmbeddings(), persist_dir=str(tmp_path / "db"))


@pytest.fixture
def store(empty_store, docs):
    build_index(split_documents(docs, chunk_size=200, chunk_overlap=20), empty_store)
    return empty_store
