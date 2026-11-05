"""Step 9: a tool-calling (ReAct-style) agent, written as an explicit loop.

LLM thinks -> picks a tool -> we run it -> result goes back -> LLM thinks again ... -> stops.
The agent only GATHERS EVIDENCE. The final answer still goes through the Step 8 grounded
generate/verify pipeline, so the agent cannot bypass citation checks.

Safety rails (all in code, none left to the LLM):
  * MAX_STEPS         hard cap on loop iterations (infinite-loop prevention)
  * repeat detection  same tool + same arguments twice -> blocked, LLM is told to change plan
  * tool errors       exceptions become an ERROR message the LLM can read and recover from
  * evidence cap      at most MAX_EVIDENCE chunks go to the final answer step
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langchain_core.tools import tool

from govtgpt import answer as A
from govtgpt.search import HybridSearcher, load_chunks

MAX_STEPS = 6
MAX_EVIDENCE = 12
ACTS = ["IT Act 2000", "IT Rules 2021", "DPDP Act 2023", "DPDP Rules 2025"]

AGENT_SYSTEM = (
    "You research Indian IT and data-protection law using tools. You never answer from memory.\n"
    f"Available acts: {', '.join(ACTS)}.\n"
    "Plan: if the question has several parts or compares two laws, search each part separately.\n"
    "Use search_law for topics, get_section when the user names a specific section or rule.\n"
    "When you have enough passages, STOP calling tools and reply with the single word DONE.\n"
    "If the tools return nothing relevant, reply DONE as well. Do not keep searching forever."
)


@dataclass
class Trace:
    question: str
    steps: list[dict] = field(default_factory=list)       # what the agent did, in order
    stopped_because: str = ""
    evidence: list[Document] = field(default_factory=list)
    result: A.Result | None = None


def _fmt(docs: list[Document]) -> str:
    return "\n\n".join(f"- {d.metadata['short']} {d.metadata['unit']} {d.metadata['section']} "
                       f"[{d.metadata['status']}]: {d.page_content[:350]}" for d in docs)


def build_tools(searcher: HybridSearcher, evidence: list[Document], fail: dict | None = None):
    """Tools close over `evidence` so everything the agent reads is collected for Step 8."""
    chunks = load_chunks()
    fail = fail or {}

    def _remember(docs):
        have = {(d.metadata["short"], d.metadata["section"], d.metadata["chunk_index"]) for d in evidence}
        for d in docs:
            k = (d.metadata["short"], d.metadata["section"], d.metadata["chunk_index"])
            if k not in have:
                evidence.append(d)
                have.add(k)

    @tool
    def search_law(query: str, act: str = "") -> str:
        """Semantic search over the Indian IT/data laws. Use for questions about a topic.
        `act` is optional: one of IT Act 2000, IT Rules 2021, DPDP Act 2023, DPDP Rules 2025."""
        if fail.get("search_law"):
            fail["search_law"] -= 1
            raise RuntimeError("503 search backend unavailable")
        flt = {"short": [act]} if act in ACTS else None
        docs = searcher.retrieve(query, k=4, filters=flt)
        _remember(docs)
        return _fmt(docs) or "No results."

    @tool
    def get_section(act: str, section: str) -> str:
        """Fetch the exact text of one section/rule, e.g. act='IT Act 2000', section='66E'."""
        hits = [d for d in chunks if d.metadata["short"] == act
                and d.metadata["section"].upper() == section.upper().strip()]
        if not hits:
            known = sorted({d.metadata["section"] for d in chunks if d.metadata["short"] == act})
            if act not in ACTS:
                return f"ERROR: unknown act '{act}'. Valid acts: {ACTS}"
            return f"ERROR: {act} has no section '{section}'. Sections present: {known[:12]}..."
        _remember(hits)
        return _fmt(hits)

    return [search_law, get_section]


def _llm_with_tools(tools):
    import os
    from langchain_google_genai import ChatGoogleGenerativeAI
    return ChatGoogleGenerativeAI(model=os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite"),
                                  temperature=0, max_retries=3).bind_tools(tools)


def run_agent(question: str, searcher: HybridSearcher, *, max_steps: int = MAX_STEPS,
              fail: dict | None = None, use_judge: bool = False) -> Trace:
    trace = Trace(question)
    tools = build_tools(searcher, trace.evidence, fail)
    by_name = {t.name: t for t in tools}
    llm = _llm_with_tools(tools)
    messages = [SystemMessage(AGENT_SYSTEM), HumanMessage(question)]
    seen_calls: set[str] = set()

    for step in range(1, max_steps + 1):
        ai: AIMessage = llm.invoke(messages)
        messages.append(ai)
        if not ai.tool_calls:                         # LLM says it is done
            trace.stopped_because = "agent finished (DONE)"
            break
        for call in ai.tool_calls:
            name, args = call["name"], call["args"]
            key = name + json.dumps(args, sort_keys=True)
            if key in seen_calls:                     # loop guard: identical repeat call
                out = "BLOCKED: you already made this exact call. Change the query or reply DONE."
                status = "blocked-repeat"
            else:
                seen_calls.add(key)
                try:
                    out = by_name[name].invoke(args)
                    status = "error-from-tool" if str(out).startswith("ERROR") else "ok"
                except Exception as e:                # tool failure -> LLM sees it, can retry
                    out = f"ERROR: tool {name} failed: {e}"
                    status = "exception"
            trace.steps.append({"step": step, "tool": name, "args": args, "status": status,
                                "chars": len(str(out))})
            messages.append(ToolMessage(content=str(out), tool_call_id=call["id"]))
    else:
        trace.stopped_because = f"MAX_STEPS={max_steps} reached (loop guard)"

    # Final answer: grounded generate + verify over everything the tools returned
    docs = trace.evidence[:MAX_EVIDENCE]
    trace.result = (A.answer(question, docs, use_judge=use_judge) if docs else
                    A.Result(question, False, A.NOT_FOUND, refused_by="no-evidence"))
    return trace
