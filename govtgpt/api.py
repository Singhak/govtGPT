"""Step 12: FastAPI service around the Step 10/11 graph, with one JSON log line per request.

Run:  .venv\Scripts\python -m uvicorn govtgpt.api:app --port 8000      (ONE worker: local Qdrant locks its folder)
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from govtgpt.graph import ask, build_graph
from govtgpt.search import HybridSearcher

load_dotenv()
LOG_PATH = Path("logs/requests.jsonl")
MAX_Q_CHARS = 500

req_log = logging.getLogger("govtgpt.requests")


def _setup_logging() -> None:
    LOG_PATH.parent.mkdir(exist_ok=True)
    if not req_log.handlers:
        h = logging.FileHandler(LOG_PATH, encoding="utf-8")
        h.setFormatter(logging.Formatter("%(message)s"))      # message is already JSON
        req_log.addHandler(h)
        req_log.setLevel(logging.INFO)
        req_log.propagate = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    _setup_logging()
    app.state.graph = build_graph(HybridSearcher())          # built once, reused by every request
    yield


app = FastAPI(title="govtGPT", lifespan=lifespan)


class AskIn(BaseModel):
    question: str = Field(min_length=3, max_length=MAX_Q_CHARS)
    thread_id: str = Field(default="default", max_length=64, pattern=r"^[A-Za-z0-9_-]+$")


class CitationOut(BaseModel):
    act: str
    section: str
    status: str
    quote: str


class AskOut(BaseModel):
    request_id: str
    answerable: bool
    answer: str
    citations: list[CitationOut]
    warnings: list[str]
    refused_by: str
    latency_ms: int


def _is_transient(e: Exception) -> bool:
    s = str(e)
    return any(k in s for k in ("RESOURCE_EXHAUSTED", "UNAVAILABLE", "429", "503", "DEADLINE"))


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask", response_model=AskOut)
def ask_endpoint(body: AskIn):       # plain def: FastAPI runs it in a worker thread (LLM calls block)
    rid = uuid.uuid4().hex[:12]
    t0 = time.perf_counter()
    rec = {"request_id": rid, "thread_id": body.thread_id, "question": body.question}
    try:
        st = ask(app.state.graph, body.question, body.thread_id)
    except Exception as e:
        rec.update(error=f"{type(e).__name__}: {str(e)[:300]}", latency_ms=int((time.perf_counter() - t0) * 1000))
        req_log.error(json.dumps(rec, ensure_ascii=False))
        if _is_transient(e):
            raise HTTPException(503, f"Model temporarily unavailable, retry later (request_id={rid})")
        raise HTTPException(500, f"Internal error (request_id={rid})")

    r = st["result"]
    ms = int((time.perf_counter() - t0) * 1000)
    cites = []
    for c in r.citations:
        m = r.sources[c.passage - 1].metadata
        cites.append(CitationOut(act=m["short"], section=str(m["section"]), status=m["status"], quote=c.quote))
    rec.update(
        standalone_question=st.get("question"), queries=st.get("queries"),
        retrieved=[f"{d.metadata['short']} {d.metadata['section']}" for d in st.get("evidence", [])],
        cited=[f"{c.act} {c.section}" for c in cites], answerable=r.answerable, refused_by=r.refused_by,
        warnings=r.warnings, retries=st.get("retries", 0), fixes=st.get("fixes", 0), latency_ms=ms,
    )
    req_log.info(json.dumps(rec, ensure_ascii=False))
    return AskOut(request_id=rid, answerable=r.answerable, answer=r.answer, citations=cites,
                  warnings=r.warnings, refused_by=r.refused_by, latency_ms=ms)
