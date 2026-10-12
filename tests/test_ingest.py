import pytest
from langchain_core.documents import Document

from rag.ingest import build_index, load_documents, split_documents


def test_load_text_and_markdown(tmp_path):
    for name in ("a.txt", "b.md"):
        (tmp_path / name).write_text("hello world", encoding="utf-8")
        [doc] = load_documents(str(tmp_path / name))
        assert doc.page_content == "hello world"
        assert doc.metadata["source"].endswith(name)


def test_load_sample_pdf_has_pages():
    pages = load_documents("documents/ps1.pdf")
    assert len(pages) == 9
    assert pages[0].metadata["page"] == 0


def test_load_errors(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_documents(str(tmp_path / "missing.pdf"))
    with pytest.raises(FileNotFoundError):
        load_documents(str(tmp_path))  # a directory is not a file
    (tmp_path / "x.csv").write_text("a,b")
    with pytest.raises(ValueError, match="Unsupported"):
        load_documents(str(tmp_path / "x.csv"))


def test_split_respects_size_and_assigns_stable_ids():
    doc = Document("word " * 300, metadata={"source": "s.txt", "page": 2})
    chunks = split_documents([doc], chunk_size=100, chunk_overlap=10)
    assert len(chunks) > 1
    assert all(len(c.page_content) <= 100 for c in chunks)
    assert [c.metadata["chunk"] for c in chunks] == list(range(len(chunks)))
    assert chunks[0].id == "s.txt#p2:c0"
    assert chunks[0].metadata["page"] == 2
    assert [c.id for c in split_documents([doc], 100, 10)] == [c.id for c in chunks]


def test_reingest_is_idempotent(empty_store, docs):
    chunks = split_documents(docs, chunk_size=200, chunk_overlap=20)
    assert build_index(chunks, empty_store) == len(chunks)
    assert build_index(chunks, empty_store) == len(chunks)  # rebuild
    assert build_index(chunks, empty_store, reset=False) == len(chunks)  # upsert, no duplicates
    extra = split_documents([Document("new text", metadata={"source": "new.txt"})])
    assert build_index(extra, empty_store, reset=False) == len(chunks) + 1
