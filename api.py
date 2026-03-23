"""
FastAPI REST API — sync ask, SSE streaming, vector store stats.

Run:
    uvicorn api:app --reload

Endpoints:
    GET  /health      — liveness check
    GET  /stats       — vector store info
    POST /ask         — blocking Q&A with sources
    POST /stream      — streaming Q&A (Server-Sent Events)
"""

import os
import json
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from rag import chain, config

load_dotenv()

app = FastAPI(
    title="RAG Q&A API",
    description="Retrieval-Augmented Generation powered by Grok.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.on_event("startup")
def startup() -> None:
    if not os.path.exists(config.PERSIST_DIR):
        raise RuntimeError(
            f"Vector store not found at '{config.PERSIST_DIR}'. "
            "Run 'python create_database.py' first."
        )


# ── Schemas ──────────────────────────────────────────────────────────────────

class QuestionRequest(BaseModel):
    question: str


class SourceSchema(BaseModel):
    content: str
    source: str
    page: int | None


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceSchema]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _doc_to_schema(doc) -> SourceSchema:
    return SourceSchema(
        content=doc.page_content[:400],
        source=doc.metadata.get("source", ""),
        page=doc.metadata.get("page"),
    )


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", tags=["Meta"])
def health():
    return {"status": "ok"}


@app.get("/stats", tags=["Meta"])
def stats():
    return chain.vectorstore_stats()


@app.post("/ask", response_model=AnswerResponse, tags=["Q&A"])
def ask(req: QuestionRequest):
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")
    docs = chain.retrieve(req.question)
    ans = chain.answer(req.question, docs)
    return AnswerResponse(
        question=req.question,
        answer=ans,
        sources=[_doc_to_schema(d) for d in docs],
    )


@app.post("/stream", tags=["Q&A"])
def stream(req: QuestionRequest):
    """
    Server-Sent Events stream.

    Each event is a JSON object:
      - `{ "token": "..." }`  — answer token
      - `{ "sources": [...], "done": true }`  — final event with source docs
    """
    if not req.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    docs = chain.retrieve(req.question)

    def generate():
        for token in chain.stream_answer(req.question, docs):
            yield f"data: {json.dumps({'token': token})}\n\n"
        sources = [_doc_to_schema(d).model_dump() for d in docs]
        yield f"data: {json.dumps({'sources': sources, 'done': True})}\n\n"

    return StreamingResponse(generate(), media_type="text/event-stream")
