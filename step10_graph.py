"""Step 10: LangGraph. Does graph (plan + self-correction + memory) beat one-shot RAG?

Run:  .venv\\Scripts\\python -W ignore step10_graph.py
Tests:
  A. 16 hard in-scope + 7 out-of-scope questions: one-shot RAG vs graph (answered / refused)
  B. memory: follow-up question that only makes sense with the previous turn (same thread_id)
"""
import json
import sys
import time

from dotenv import load_dotenv

from govtgpt.answer import answer
from govtgpt.graph import ask, build_graph
from govtgpt.search import HybridSearcher

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")
s = HybridSearcher()
g = build_graph(s)
hard = json.load(open("artifacts/eval_hard.json", encoding="utf-8"))


def retry(fn, *a):
    for _ in range(4):
        try:
            return fn(*a)
        except Exception as e:
            print("   (retry:", str(e)[:70], ")")
            time.sleep(25)
    raise RuntimeError("gave up")


def gold_ok(r, gold):
    cited = {(r.sources[c.passage - 1].metadata["short"], r.sources[c.passage - 1].metadata["section"])
             for c in r.citations}
    return bool(cited & {tuple(x) for x in gold})


print("=== A1. in-scope hard questions (expect answered)")
tally = {"one": [0, 0], "graph": [0, 0]}      # [answered, answered AND cites a gold section]
for i, item in enumerate(hard["in_scope"]):
    q = item["q"]
    r1 = retry(lambda: answer(q, s.retrieve(q, k=6)))
    st = retry(ask, g, q, f"hard-{i}")
    r2 = st["result"]
    tally["one"][0] += r1.answerable
    tally["one"][1] += r1.answerable and gold_ok(r1, item["gold"])
    tally["graph"][0] += r2.answerable
    tally["graph"][1] += r2.answerable and gold_ok(r2, item["gold"])
    flag = "" if r1.answerable == r2.answerable else "  <-- DIFFERENT"
    print(f"\n[{i}] {q}\n  one-shot: answerable={r1.answerable} refused_by={r1.refused_by!r}"
          f" | graph: answerable={r2.answerable} refused_by={r2.refused_by!r}{flag}")
    for line in st["log"]:
        print("     ", line)

print("\n=== A2. out-of-scope (expect refused)")
oos = {"one": 0, "graph": 0}
for i, q in enumerate(hard["out_of_scope"]):
    r1 = retry(lambda: answer(q, s.retrieve(q, k=6)))
    st = retry(ask, g, q, f"oos-{i}")
    oos["one"] += not r1.answerable
    oos["graph"] += not st["result"].answerable
    print(f"  {q[:60]!r}: one-shot refused={not r1.answerable} | graph refused={not st['result'].answerable}"
          f" | {st['log'][-1][:70]}")

n = len(hard["in_scope"])
print(f"\nSUMMARY in-scope (n={n}): answered one-shot {tally['one'][0]} vs graph {tally['graph'][0]};"
      f" answered+cites gold: {tally['one'][1]} vs {tally['graph'][1]}")
print(f"SUMMARY out-of-scope refused (n={len(hard['out_of_scope'])}): one-shot {oos['one']} vs graph {oos['graph']}")

print("\n=== B. memory (same thread_id)")
t1 = retry(ask, g, "What is the penalty for a data breach under the DPDP Act?", "mem")
print("turn 1:", t1["result"].answer[:200])
before = len(t1["log"])
t2 = retry(ask, g, "And what did the older IT Act say about the same thing?", "mem")
print("turn 2 (follow-up):", t2["result"].answer[:300])
for line in t2["log"][before:]:
    print("   ", line)
t3 = retry(ask, g, "And what did the older IT Act say about the same thing?", "fresh-thread")
print("turn 2 on a NEW thread (no memory):", t3["result"].answer[:200], "| refused_by:", t3["result"].refused_by)
