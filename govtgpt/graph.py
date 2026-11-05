"""Step 10: the same pipeline as a LangGraph StateGraph.

    (new in Step 11: refused_by='numbers' -> fix -> answer, once)
    START -> plan -> retrieve -> answer(verify) --answered--------------> finalize -> END
                        ^                      \--refused, retries left--> rewrite -+
                        +------------------------------------------------------------+
                                               \--refused, no retries----> finalize -> END

What LangGraph adds over the Step 9 hand-written loop:
  * explicit state (one typed dict that every node reads/writes)
  * conditional edges (routing is data, visible and testable, not buried in if/else)
  * checkpointer + thread_id  -> memory across turns, resume after a crash
  * recursion_limit           -> framework-level loop cap (we keep our own counter too)
Safety stays in code: the answer node is the Step 8/8b verifier, so no node can skip it.
"""
from __future__ import annotations

import operator
import os
from typing import Annotated, TypedDict

from langchain_core.documents import Document
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field

from govtgpt import answer as A
from govtgpt.search import HybridSearcher

MAX_FIXES = 1            # regenerate attempts after a numbers refusal
MAX_RETRIES = 2          # rewrite attempts after the first try
K_PER_QUERY = 5
MAX_EVIDENCE = 12


class State(TypedDict, total=False):
    question: str
    history: Annotated[list[str], operator.add]   # reducer: node outputs are APPENDED
    queries: list[str]                            # what to search for
    evidence: list[Document]
    result: A.Result
    retries: int
    fixes: int                                    # regenerate attempts after a `numbers` refusal
    feedback: str
    log: Annotated[list[str], operator.add]       # human-readable trace of node decisions


class Plan(BaseModel):
    in_scope: bool = Field(description="true if the question is about Indian IT Act 2000, IT Rules 2021, "
                                       "DPDP Act 2023 or DPDP Rules 2025")
    standalone_question: str = Field(description="the question rewritten so it needs no chat history "
                                                 "(resolve 'it', 'that section', 'the same law')")
    queries: list[str] = Field(description="1 to 3 short search queries; one per distinct law or sub-question")


class Rewrite(BaseModel):
    queries: list[str] = Field(description="1 to 3 NEW search queries, using different words "
                                           "(e.g. the statute's own vocabulary) than before")


def _llm(schema):
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"),
                                  temperature=0, max_retries=3).with_structured_output(schema)


def build_graph(searcher: HybridSearcher, *, use_judge: bool = False, checkpointer=None,
                fix_numbers: bool = True):
    FIX_NUMBERS = fix_numbers
    def plan(s: State) -> State:
        hist = "\n".join(s.get("history", [])[-6:]) or "(none)"
        p: Plan = _llm(Plan).invoke(
            "Plan retrieval for a legal Q&A system over Indian IT and data-protection law.\n"
            f"CHAT HISTORY:\n{hist}\n\nNEW QUESTION: {s['question']}")
        qs = p.queries[:3] if p.in_scope and p.queries else []
        return {"question": p.standalone_question, "queries": qs, "retries": 0, "fixes": 0, "feedback": "", "evidence": [],
                "log": [f"plan: in_scope={p.in_scope} queries={qs}"]}

    def route_after_plan(s: State) -> str:
        return "retrieve" if s["queries"] else "finalize"

    def retrieve(s: State) -> State:
        have = {(d.metadata["short"], d.metadata["section"], d.metadata["chunk_index"])
                for d in s.get("evidence", [])}
        ev = list(s.get("evidence", []))
        for q in s["queries"]:
            for d in searcher.retrieve(q, k=K_PER_QUERY):
                key = (d.metadata["short"], d.metadata["section"], d.metadata["chunk_index"])
                if key not in have:
                    have.add(key)
                    ev.append(d)
        return {"evidence": ev[:MAX_EVIDENCE], "log": [f"retrieve: {len(ev[:MAX_EVIDENCE])} chunks"]}

    def answer_node(s: State) -> State:
        d = A.generate(s["question"], s["evidence"], s.get("feedback", ""))
        r = A.verify(s["question"], s["evidence"], d, use_judge=use_judge)
        return {"result": r, "log": [f"answer: answerable={r.answerable} refused_by={r.refused_by!r}"]}

    def route_after_answer(s: State) -> str:
        r = s["result"]
        if r.answerable:
            return "finalize"
        # retry only when a different search could help: retrieval-side miss, not a failed check
        if s["retries"] < MAX_RETRIES and r.refused_by in ("llm", "quote"):
            return "rewrite"
        # generation-side failure: same evidence, regenerate with feedback on what was wrong
        if FIX_NUMBERS and s.get("fixes", 0) < MAX_FIXES and r.refused_by == "numbers":
            return "fix"
        return "finalize"

    def rewrite(s: State) -> State:
        rw: Rewrite = _llm(Rewrite).invoke(
            "The previous search did not retrieve passages that answer the question. Write new "
            "search queries using different wording (the statute's own terms, synonyms, the "
            "specific obligation or offence).\n"
            f"QUESTION: {s['question']}\nPREVIOUS QUERIES: {s['queries']}")
        qs = rw.queries[:3] or s["queries"]
        return {"queries": qs, "retries": s["retries"] + 1, "log": [f"rewrite #{s['retries'] + 1}: {qs}"]}

    def fix(s: State) -> State:
        w = " ".join(s["result"].warnings)
        fb = (f"Your previous answer was rejected: {w} Only state numbers, amounts and time periods "
              "that appear verbatim in a passage you quote. If the exact figure is not in the "
              "passages, say what the passages do say without the figure.")
        return {"feedback": fb, "fixes": s.get("fixes", 0) + 1, "log": ["fix: regenerate with number feedback"]}

    def finalize(s: State) -> State:
        r = s.get("result") or A.Result(s["question"], False, A.NOT_FOUND, refused_by="out-of-scope")
        return {"result": r, "history": [f"Q: {s['question']}", f"A: {r.answer[:300]}"]}

    g = StateGraph(State)
    g.add_node("plan", plan)
    g.add_node("retrieve", retrieve)
    g.add_node("answer", answer_node)
    g.add_node("rewrite", rewrite)
    g.add_node("fix", fix)
    g.add_node("finalize", finalize)
    g.add_edge(START, "plan")
    g.add_conditional_edges("plan", route_after_plan, {"retrieve": "retrieve", "finalize": "finalize"})
    g.add_edge("retrieve", "answer")
    g.add_conditional_edges("answer", route_after_answer, {"rewrite": "rewrite", "fix": "fix", "finalize": "finalize"})
    g.add_edge("fix", "answer")
    g.add_edge("rewrite", "retrieve")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer or MemorySaver())


def ask(graph, question: str, thread_id: str = "default") -> State:
    cfg = {"configurable": {"thread_id": thread_id}, "recursion_limit": 25}
    return graph.invoke({"question": question, "log": []}, cfg)
