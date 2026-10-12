# documentloader

Ask questions about your documents and get answers grounded in, and cited to, the retrieved pages.

LangChain, ChromaDB, OpenAI embeddings, xAI Grok. CLI, Streamlit and FastAPI front ends over one small core.

## How it works

```
Documents (PDF / TXT / URL)
        │
        ▼
   split into chunks
        │
        ▼
  embed (OpenAI)  ──►  ChromaDB (persisted)
                               │
User question ──► embed ──► retrieve top-k
                               │
                          Grok LLM  ──►  Streaming answer + source citations
```

Each retrieved chunk is shown to the model as `[Source N: file, page P]`, and the same numbering is returned with the answer, so "Source 2" in the text maps to the second citation.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env    # set XAI_API_KEY and OPENAI_API_KEY
```

## Usage

Build the index (rebuilds by default; `--append` adds to it):

```bash
python create_database.py                      # documents/ps1.pdf
python create_database.py documents/*.pdf notes.md
python create_database.py https://en.wikipedia.org/wiki/Retrieval-augmented_generation
```

Then ask questions from any of the three interfaces:

```bash
python main.py              # terminal chat; /clear resets memory, /exit quits
streamlit run app.py        # browser chat
uvicorn api:app --reload    # REST API, docs at http://localhost:8000/docs
```

| Method | Endpoint | |
|---|---|---|
| GET | `/health` | liveness |
| GET | `/ready` | 503 until keys are set and the store has chunks |
| GET | `/stats` | vector store info |
| POST | `/ask` | answer with sources |
| POST | `/stream` | answer as Server-Sent Events |

```bash
curl -X POST localhost:8000/ask -H "Content-Type: application/json" \
  -d '{"question": "What is due and when?", "k": 4, "history": []}'
```

`/stream` sends `{"token": ...}` events, then `{"sources": [...], "done": true}`. Docker: `docker compose up` (mounts `./chroma_db` and `./documents`).

Settings come from the environment or `.env`: `MODEL`, `EMBEDDING_MODEL`, `CHUNK_SIZE` (1000), `CHUNK_OVERLAP` (200), `RETRIEVAL_K` (4), `HISTORY_TURNS` (5), `PERSIST_DIR` (`./chroma_db`).

## Tests

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

The tests run offline: a deterministic hashing embedder over a temporary Chroma store, and a fake LLM. They cover chunking, retrieval, prompt and citation assembly, conversation memory, the API (including SSE) and error paths.

## Retrieval eval

`eval/questions.jsonl` has 14 questions about the sample document, each labelled with the page that answers it. The script indexes the document into a temporary store and checks where that page first appears in the top-k chunks.

```bash
python -m eval.run_eval -v                    # offline hashing embedder
python -m eval.run_eval --embedder openai     # text-embedding-3-small, needs OPENAI_API_KEY
```

Offline baseline (hashing embedder, chunk 1000 / overlap 200, k=4):

| hit@1 | hit@4 | MRR@4 |
|---|---|---|
| 0.500 | 0.786 | 0.619 |

This is a lexical floor, not the configured system. The OpenAI embedder has not been measured here.

## Layout

```
rag/
  config.py          settings and validation
  ingest.py          load, split, index
  chain.py           retrieve / answer / stream_answer / format_sources
  embeddings.py      OpenAI embeddings and the offline hashing embedder
create_database.py   ingestion CLI
main.py              terminal chat
app.py               Streamlit chat
api.py               FastAPI + SSE
eval/                retrieval questions and hit@k / MRR script
tests/               offline pytest suite
documents/ps1.pdf    sample document (CS229 problem set 1)
```
