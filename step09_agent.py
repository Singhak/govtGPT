"""Step 9: tool-calling agent. Shows planning, tool use, tool failure, loop guard.

Run:  .venv\\Scripts\\python -W ignore step09_agent.py
"""
import sys
import time

from dotenv import load_dotenv

from govtgpt.agent import run_agent
from govtgpt.answer import answer
from govtgpt.search import HybridSearcher

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")
s = HybridSearcher()


def show(title, q, **kw):
    print(f"\n{'=' * 90}\n{title}\nQ: {q}")
    t = None
    for attempt in range(3):
        try:
            t = run_agent(q, s, **kw)
            break
        except Exception as e:
            print("   (retry after error:", str(e)[:80], ")")
            time.sleep(20)
    for st in t.steps:
        print(f"  step {st['step']}: {st['tool']}({st['args']}) -> {st['status']}")
    print(f"  stopped: {t.stopped_because} | evidence chunks: {len(t.evidence)}")
    r = t.result
    print(f"  answerable={r.answerable} refused_by={r.refused_by!r}")
    print("  A:", r.answer[:420])
    for w in r.warnings:
        print("  WARNING:", w[:160])
    return t


# 1) Single lookup: the agent should use get_section, not a fuzzy search
show("1. exact lookup", "What does Section 66E of the IT Act say?")

# 2) Multi-part question: needs TWO searches (one per law). Compare with plain one-shot RAG.
q2 = ("How do the IT Act 2000 (section 43A) and the DPDP Act 2023 each treat a company that "
      "fails to protect personal data?")
show("2. multi-part (agent)", q2)
r = answer(q2, s.retrieve(q2, k=6))
print(f"  [baseline one-shot RAG] answerable={r.answerable} | sections cited: "
      f"{sorted({(r.sources[c.passage-1].metadata['short'], r.sources[c.passage-1].metadata['section']) for c in r.citations})}")

# 3) Tool failure: search backend fails twice, then recovers
show("3. tool failure (search fails 2x)", "What is the penalty for a data breach under the DPDP Act?",
     fail={"search_law": 2})

# 4) Impossible request: get_section keeps failing. The loop guard must stop it.
show("4. loop guard", "Quote the exact text of Section 999 of the IT Act 2000, I need it word for word.",
     max_steps=4)

# 5) Out of scope: should stop quickly and refuse
show("5. out of scope", "What is the fine under the California CCPA?")
