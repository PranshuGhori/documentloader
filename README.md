# RAG Q&A — Powered by LLM

A production-ready **Retrieval-Augmented Generation** pipeline. Ask questions about your documents and get grounded, cited answers — with streaming, conversation memory, and three interfaces to choose from.

**Stack:** LangChain · xAI Grok · ChromaDB · OpenAI Embeddings · FastAPI · Streamlit

---

## How It Works

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

---

## Setup

### 1. Clone & create a virtual environment

```bash
git clone https://github.com/PranshuGhori/documentloader.git
cd documentloader

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
```

### 2. Install dependencies

```bash
pip install -r requirement.txt
```

### 3. Configure environment variables

```bash
cp .env.example .env
```

Edit `.env`:

```env
XAI_API_KEY=your-xai-api-key
OPENAI_API_KEY=your-openai-api-key
```

| Variable | Purpose |
|---|---|
| `XAI_API_KEY` | Grok LLM (answer generation) |
| `OPENAI_API_KEY` | OpenAI `text-embedding-3-small` (retrieval) |

---

## Usage

### Step 1 — Ingest documents

```bash
# Default (documents/ps1.pdf)
python create_database.py

# Any PDF
python create_database.py path/to/file.pdf

# Multiple files / glob
python create_database.py documents/*.pdf

# Web URL
python create_database.py https://en.wikipedia.org/wiki/Retrieval-augmented_generation
```

Only needed once — or whenever your documents change.

### Step 2 — Ask questions

**CLI** — streaming output, source table, conversation memory

```bash
python main.py
```

```
╭─────────────────────────────────────────╮
│ RAG Q&A  ·  powered by grok-...         │
│ 42 chunks indexed · top-4 retrieval     │
╰─────────────────────────────────────────╯

You: What is this document about?

Assistant
The document covers...

 #  File       Page  Excerpt
 1  ps1.pdf       1  The assignment requires...
 2  ps1.pdf       3  Students should submit...
```

**Streamlit** — chat UI in your browser

```bash
streamlit run app.py
```

- Streaming token-by-token output
- Expandable source citations under each answer
- Conversation memory across turns
- Sidebar with vector store stats

**FastAPI** — REST API at `http://localhost:8000`

```bash
uvicorn api:app --reload
```

| Method | Endpoint | Description |
|---|---|---|
| GET | `/health` | Liveness check |
| GET | `/stats` | Vector store info |
| POST | `/ask` | Blocking Q&A with sources |
| POST | `/stream` | SSE streaming Q&A |

```bash
# Blocking
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is this document about?"}'

# Streaming (Server-Sent Events)
curl -N -X POST http://localhost:8000/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "Summarise the key points."}'
```

Interactive API docs: `http://localhost:8000/docs`

**Docker**

```bash
docker-compose up
```

---

## Project Structure

```
.
├── rag/                     # Shared core package
│   ├── config.py            # All settings (models, paths, chunk size)
│   ├── chain.py             # retrieve / answer / stream_answer / stats
│   └── ingest.py            # Multi-format document loader
│
├── create_database.py       # Ingestion CLI (PDF, TXT, URL, glob)
├── main.py                  # Rich terminal interface
├── api.py                   # FastAPI REST + SSE streaming
├── app.py                   # Streamlit chat UI
│
├── documents/               # Drop your source files here
│   └── ps1.pdf
├── chroma_db/               # Persisted vector store (gitignored)
│
├── Dockerfile
├── docker-compose.yml
├── requirement.txt
├── .env.example
└── .env                     # API keys (gitignored)
```

---

## Configuration

All tunable via environment variables (or `.env`):

| Variable | Default | Description |
|---|---|---|
| `MODEL` | `grok-4-fast-non-reasoning` | Grok model for generation |
| `CHUNK_SIZE` | `1000` | Characters per chunk |
| `CHUNK_OVERLAP` | `200` | Overlap between chunks |
| `RETRIEVAL_K` | `4` | Number of chunks retrieved per query |
| `PERSIST_DIR` | `./chroma_db` | Vector store location |
