"""Step 4: compare dense vs BM25 vs hybrid on the same questions.

Run:  .venv\Scripts\python -W ignore step04_hybrid_demo.py ["your question"]
"""
import sys

from govtgpt.search import HybridSearcher

sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS = sys.argv[1:] or [
    "What is Section 43A?",
    "Can I be arrested for posting an offensive message online?",
    "Within how many hours must an intermediary remove non-consensual intimate images?",
    "What are the duties of a Data Fiduciary?",
]
s = HybridSearcher()


def show(label, docs, n=4):
    print(f"  {label}")
    for d in docs[:n]:
        m = d.metadata
        print(f"     {m['short']} {m['unit']} {m['section']} (chunk {m['chunk_index']}) "
              f"[{m['status']}]")


for q in QUESTIONS:
    print(f"\n=== Q: {q}")
    show("DENSE ", s.dense(q))
    show("BM25  ", s.sparse(q))
    show("HYBRID", s.hybrid(q))
