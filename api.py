"""
FastAPI REST API — sync ask, SSE streaming, vector store stats.

Run:
    uvicorn api:app --reload

Endpoints:
    GET  /health      — liveness check
    GET  /ready       — readiness: keys configured and vector store non-empty
    GET  /stats       — vector store info
    POST /ask         — blocking Q&A with sources
    POST /stream      — streaming Q&A (Server-Sent Events)
"""

import json
import logging
from typing import Literal

from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from langchain_core.messages import AIMessage, HumanMessage
from pydantic import BaseModel, Field, field_validator

from rag import chain, config

log = logging.getLogger("api")

app = FastAPI(
    title="RAG Q&A API",
    description="Retrieval-Augmented Generation powered by Grok.",
    version="1.1.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Dependencies (overridden in tests) ───────────────────────────────────────


def get_vectorstore():
    return chain.default_vectorstore()


def get_llm():
    return chain.default_llm()


@app.exception_handler(config.ConfigError)
@app.exception_handler(chain.EmptyStoreError)
def _unavailable(request: Request, exc: Exception):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


# ── Schemas ──────────────────────────────────────────────────────────────────


class Turn(BaseModel):
    role: Literal["user", "assistant"]
    content: str = Field(max_length=8000)


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=40)
    k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def _not_blank(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("question cannot be blank")
        return v.strip()

    def messages(self):
        return [HumanMessage(t.content) if t.role == "user" else AIMessage(t.content) for t in self.history]


class SourceSchema(BaseModel):
    index: int
    source: str
    page: int | None
    content: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceSchema]


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


# ── Routes ───────────────────────────────────────────────────────────────────


@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok"}


@app.get("/ready", tags=["Meta"])
def ready(vs=Depends(get_vectorstore), llm=Depends(get_llm)):
    chunks = vs._collection.count()
    if chunks == 0:
        raise chain.EmptyStoreError("The vector store is empty. Run 'python create_database.py' first.")
    return {"status": "ready", "chunks": chunks}


@app.get("/stats", tags=["Meta"])
def stats(vs=Depends(get_vectorstore)):
    return chain.vectorstore_stats(vs)


@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
def ask(req: QuestionRequest, vs=Depends(get_vectorstore), llm=Depends(get_llm)):
    docs = chain.retrieve(req.question, vs, k=req.k)
    try:
        ans = chain.answer(req.question, docs, req.messages(), llm=llm)
    except Exception as e:
        log.exception("LLM call failed")
        return JSONResponse(status_code=502, content={"detail": f"LLM request failed: {e}"})
    return AnswerResponse(question=req.question, answer=ans, sources=chain.format_sources(docs))


@app.post("/stream", tags=["Q&A"])
def stream(req: QuestionRequest, vs=Depends(get_vectorstore), llm=Depends(get_llm)):
    """
    Server-Sent Events stream.

    Each event is a JSON object:
      - `{ "token": "..." }`                 — answer token
      - `{ "sources": [...], "done": true }` — final event with source docs
      - `{ "error": "...", "done": true }`   — the LLM failed mid-stream
    """
    docs = chain.retrieve(req.question, vs, k=req.k)
    history = req.messages()

    def generate():
        try:
            for token in chain.stream_answer(req.question, docs, history, llm=llm):
                yield _sse({"token": token})
        except Exception as e:
            log.exception("LLM stream failed")
            yield _sse({"error": f"LLM request failed: {e}", "done": True})
            return
        yield _sse({"sources": chain.format_sources(docs), "done": True})

    return StreamingResponse(generate(), media_type="text/event-stream")
