import json

import pytest
from fastapi.testclient import TestClient

import api
from rag.chain import open_vectorstore
from rag.embeddings import HashingEmbeddings
from tests.conftest import FakeLLM


@pytest.fixture
def client(store):
    llm = FakeLLM("Grounded answer.")
    api.app.dependency_overrides = {api.get_vectorstore: lambda: store, api.get_llm: lambda: llm}
    with TestClient(api.app) as c:
        c.llm = llm
        yield c
    api.app.dependency_overrides = {}


def events(response):
    return [json.loads(line[6:]) for line in response.text.splitlines() if line.startswith("data: ")]


def test_health_and_ready(client):
    assert client.get("/health").json() == {"status": "ok"}
    assert client.get("/ready").json() == {"status": "ready", "chunks": 3}
    assert client.get("/stats").json()["chunks"] == 3


def test_ask_returns_answer_and_citations(client):
    r = client.post("/ask", json={"question": "  When did the French Revolution begin?  ", "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["question"] == "When did the French Revolution begin?"
    assert body["answer"] == "Grounded answer."
    assert len(body["sources"]) == 2
    assert body["sources"][0] == {
        "index": 1,
        "source": "bio.pdf",
        "page": 3,
        "content": "The French Revolution began in 1789 with the storming of the Bastille.",
    }


def test_ask_passes_history(client):
    history = [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]
    client.post("/ask", json={"question": "and?", "history": history})
    assert [m.content for m in client.llm.calls[0][1:]] == ["hi", "hello", "and?"]


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {"question": ""},
        {"question": "   "},
        {"question": "x" * 2001},
        {"question": "ok", "k": 0},
        {"question": "ok", "k": 21},
        {"question": "ok", "history": [{"role": "system", "content": "x"}]},
    ],
)
def test_ask_rejects_bad_input(client, payload):
    assert client.post("/ask", json=payload).status_code == 422


def test_stream_sse(client):
    with client.stream("POST", "/stream", json={"question": "photosynthesis"}) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("text/event-stream")
        r.read()
    evs = events(r)
    assert "".join(e["token"] for e in evs[:-1]) == "Grounded answer. "
    assert evs[-1]["done"] is True
    assert evs[-1]["sources"][0]["page"] == 2


def test_stream_llm_failure_emits_error_event(client):
    client.llm.fail = True
    evs = events(client.post("/stream", json={"question": "q"}))
    assert evs[-1]["done"] is True and "upstream down" in evs[-1]["error"]


def test_ask_llm_failure_is_502(client):
    client.llm.fail = True
    r = client.post("/ask", json={"question": "q"})
    assert r.status_code == 502 and "upstream down" in r.json()["detail"]


def test_empty_store_is_503(client, tmp_path):
    empty = open_vectorstore(HashingEmbeddings(), persist_dir=str(tmp_path / "empty"))
    api.app.dependency_overrides[api.get_vectorstore] = lambda: empty
    for r in (
        client.post("/ask", json={"question": "q"}),
        client.post("/stream", json={"question": "q"}),
        client.get("/ready"),
    ):
        assert r.status_code == 503
        assert "create_database.py" in r.json()["detail"]


def test_missing_keys_is_503(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    api.chain.default_vectorstore.cache_clear()
    with TestClient(api.app) as c:
        r = c.post("/ask", json={"question": "q"})
        assert r.status_code == 503 and "OPENAI_API_KEY" in r.json()["detail"]
        assert c.get("/health").status_code == 200
