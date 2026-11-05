"""Step 8: grounded answers. Tests: in-scope, struck-down, and 7 out-of-scope questions.

Run:  .venv\Scripts\python -W ignore step08_answer.py
"""
import json
import sys

from dotenv import load_dotenv

from govtgpt.answer import answer
from govtgpt.search import HybridSearcher

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")
s = HybridSearcher()

hard = json.load(open("artifacts/eval_hard.json", encoding="utf-8"))
in_scope = [
    "What is the penalty for failing to protect personal data under the DPDP Act?",
    "Within how many hours must an intermediary remove non-consensual intimate images?",
    "What is the punishment for sending offensive messages through a computer under section 66A?",
    "Can a Data Fiduciary process children's data without parental consent?",
]
out_scope = hard["out_of_scope"]


def run(q):
    r = answer(q, s.retrieve(q, k=6))
    print(f"\nQ: {q}\n  answerable={r.answerable}")
    print("  A:", r.answer[:500])
    for c in r.citations:
        m = r.sources[c.passage - 1].metadata
        print(f"  cite: {m['short']} s.{m['section']} | \"{c.quote}\"")
    for w in r.warnings:
        print("  WARNING:", w)
    return r


print("=== IN SCOPE (expect answerable=True)")
ok_in = sum(run(q).answerable for q in in_scope)
print("\n=== OUT OF SCOPE (expect answerable=False)")
ok_out = sum(not run(q).answerable for q in out_scope)
print(f"\nSUMMARY: in-scope answered {ok_in}/{len(in_scope)} | out-of-scope refused {ok_out}/{len(out_scope)}")
