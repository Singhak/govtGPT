"""Step 4: dense + BM25 + hybrid (RRF)."""
from __future__ import annotations

import json
import re
import time
from pathlib import Path

from langchain_core.documents import Document
from pydantic import BaseModel, Field
from qdrant_client import models as qm
from rank_bm25 import BM25Okapi

from govtgpt.store import QUERY_CACHE, load_store

RRF_K = 60  # standard constant: dampens the gap between rank 1 and rank 2


def _tok(text: str) -> list[str]:
    # \w+ keeps "43a" as ONE token, so it differs from "43" (this is the point of BM25 here)
    return re.findall(r"\w+", text.lower())


def _key(d: Document) -> tuple:
    m = d.metadata
    return (m["short"], m["section"], m["chunk_index"], hash(d.page_content))


def load_chunks(path: str = "artifacts/chunks.jsonl") -> list[Document]:
    with open(Path(path), encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    return [Document(page_content=r["text"], metadata=r["metadata"]) for r in rows]


class HybridSearcher:
    def __init__(self):
        self.docs = load_chunks()
        self.bm25 = BM25Okapi([_tok(d.page_content) for d in self.docs])
        self.store = load_store()

    # ---- Step 5a: metadata filters -------------------------------------
    # filters = {"short": ["IT Act 2000"], "doc_type": ["act"],
    #            "exclude_status": ["struck_down"]}
    @staticmethod
    def _qdrant_filter(f: dict | None):
        if not f:
            return None
        must, must_not = [], []
        for key in ("short", "doc_type", "status"):
            if f.get(key):
                must.append(qm.FieldCondition(key=f"metadata.{key}",
                                              match=qm.MatchAny(any=f[key])))
        if f.get("exclude_status"):
            must_not.append(qm.FieldCondition(key="metadata.status",
                                              match=qm.MatchAny(any=f["exclude_status"])))
        return qm.Filter(must=must or None, must_not=must_not or None)

    @staticmethod
    def _passes(d: Document, f: dict | None) -> bool:
        if not f:
            return True
        m = d.metadata
        for key in ("short", "doc_type", "status"):
            if f.get(key) and m[key] not in f[key]:
                return False
        return m["status"] not in f.get("exclude_status", [])

    # Query-embedding cache (memory + disk). Free tier allows only 100 embed
    # requests/minute, and the same question is often searched several times.
    _CACHE_FILE = Path(QUERY_CACHE)

    def _embed_query(self, q: str) -> list[float]:
        if not hasattr(self, "_qcache"):
            try:
                self._qcache = json.loads(self._CACHE_FILE.read_text(encoding="utf-8"))
            except Exception:
                self._qcache = {}
        if q not in self._qcache:
            for attempt in range(5):
                try:
                    self._qcache[q] = self.store.embeddings.embed_query(q)
                    break
                except Exception as e:           # 429 rate limit -> wait, retry
                    if attempt == 4:
                        raise
                    time.sleep(20 * (attempt + 1))
            self._CACHE_FILE.parent.mkdir(exist_ok=True)
            self._CACHE_FILE.write_text(json.dumps(self._qcache), encoding="utf-8")
        return self._qcache[q]

    def dense(self, q: str, k: int = 20, filters: dict | None = None) -> list[Document]:
        return self.store.similarity_search_by_vector(
            self._embed_query(q), k=k, filter=self._qdrant_filter(filters))

    def sparse(self, q: str, k: int = 20, filters: dict | None = None) -> list[Document]:
        scores = self.bm25.get_scores(_tok(q))
        order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        out = [self.docs[i] for i in order
               if scores[i] > 0 and self._passes(self.docs[i], filters)]
        return out[:k]

    def retrieve(self, q: str, k: int = 5, filters: dict | None = None) -> list[Document]:
        """Default retriever. DENSE, because it won on every question set we measured
        (see Step 7 in the challenges file). hybrid() is kept for later re-testing."""
        return self.dense(q, k, filters)

    def hybrid(self, q: str, k: int = 5, pool: int = 20,
               filters: dict | None = None) -> list[Document]:
        """Reciprocal Rank Fusion: score(d) = sum over lists of 1 / (RRF_K + rank)."""
        fused: dict[tuple, float] = {}
        by_key: dict[tuple, Document] = {}
        for ranked in (self.dense(q, pool, filters), self.sparse(q, pool, filters)):
            for rank, d in enumerate(ranked, start=1):
                key = _key(d)
                by_key[key] = d
                fused[key] = fused.get(key, 0.0) + 1.0 / (RRF_K + rank)
        top = sorted(fused, key=fused.get, reverse=True)[:k]
        return [by_key[x] for x in top]


# ---- Step 5b: reranking (LLM reads question + candidates together) -------
class _Score(BaseModel):
    id: int = Field(description="candidate number as given")
    relevance: int = Field(description="0-10: does this chunk directly answer the question?")


class _Ranking(BaseModel):
    scores: list[_Score]


def rerank(question: str, docs: list[Document], top_n: int = 5,
           snippet_chars: int = 900) -> list[Document]:
    """One LLM call scores every candidate; we keep the best top_n.

    The embedding sees question and chunk SEPARATELY. A reranker reads them
    TOGETHER, so it can judge 'does this chunk actually answer the question?'.
    """
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI

    llm = ChatGoogleGenerativeAI(
        model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"),
        temperature=0, max_retries=2,
    ).with_structured_output(_Ranking)
    NL = chr(10)
    listing = (NL + NL).join(f"[{i}] {d.page_content[:snippet_chars]}"
                             for i, d in enumerate(docs))
    prompt = (
        "You are a legal retrieval judge. For each candidate passage give a relevance "
        "score 0-10 for how directly it answers the question. Judge only the text."
        + NL + NL + f"QUESTION: {question}" + NL + NL + "CANDIDATES:" + NL + listing
    )
    result = llm.invoke(prompt)
    score = {s.id: s.relevance for s in result.scores}
    order = sorted(range(len(docs)), key=lambda i: score.get(i, 0), reverse=True)
    return [docs[i] for i in order[:top_n]]
