import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from rag import chain, config
from tests.conftest import FakeLLM


def test_retrieve_ranks_relevant_page_first(store):
    docs = chain.retrieve("When did the French Revolution begin?", store, k=2)
    assert len(docs) == 2
    assert docs[0].metadata["page"] == 2


def test_retrieve_empty_store_raises(empty_store):
    with pytest.raises(chain.EmptyStoreError):
        chain.retrieve("anything", empty_store)


def test_stats(store):
    assert chain.vectorstore_stats(store)["chunks"] == 3


def test_prompt_has_numbered_citations(docs):
    llm = FakeLLM()
    assert chain.answer("What makes ATP?", docs[:2], llm=llm) == "The answer, per Source 1."
    system, human = llm.calls[0]
    assert isinstance(system, SystemMessage)
    assert "[Source 1: bio.pdf, page 1]\nThe mitochondria" in system.content
    assert "[Source 2: bio.pdf, page 2]" in system.content
    assert human.content == "What makes ATP?"


def test_no_docs_still_answers():
    llm = FakeLLM()
    chain.answer("q", [], llm=llm)
    assert "(no relevant context found)" in llm.calls[0][0].content


def test_history_goes_between_system_and_question(docs):
    llm = FakeLLM()
    history = [HumanMessage("first q"), AIMessage("first a")]
    chain.answer("follow-up", docs, history, llm=llm)
    assert [m.content for m in llm.calls[0][1:]] == ["first q", "first a", "follow-up"]


def test_braces_in_history_are_not_templated(docs):
    llm = FakeLLM()
    chain.answer("{context}", docs, [HumanMessage("{question}"), AIMessage("ok")], llm=llm)
    assert [m.content for m in llm.calls[0][1:]] == ["{question}", "ok", "{context}"]


def test_trim_history():
    history = [HumanMessage(str(i)) if i % 2 == 0 else AIMessage(str(i)) for i in range(10)]
    assert [m.content for m in chain.trim_history(history, turns=2)] == ["6", "7", "8", "9"]
    assert chain.trim_history(history, turns=0) == []
    assert len(chain.trim_history(history)) == 2 * config.HISTORY_TURNS


def test_stream_answer_yields_tokens(docs):
    assert "".join(chain.stream_answer("q", docs, llm=FakeLLM("a b c"))) == "a b c "


def test_format_sources():
    long = Document("x" * 500, metadata={"source": "/abs/path/file.pdf", "page": 0})
    web = Document("short", metadata={"source": "https://example.com"})
    s1, s2 = chain.format_sources([long, web], excerpt=100)
    assert s1 == {"index": 1, "source": "file.pdf", "page": 1, "content": "x" * 100 + "…"}
    assert s2["page"] is None and s2["content"] == "short"


def test_missing_keys_raise_config_error(monkeypatch):
    monkeypatch.delenv("XAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    chain.default_llm.cache_clear()
    chain.default_vectorstore.cache_clear()
    with pytest.raises(config.ConfigError, match="XAI_API_KEY"):
        chain.default_llm()
    with pytest.raises(config.ConfigError, match="OPENAI_API_KEY"):
        chain.default_vectorstore()
