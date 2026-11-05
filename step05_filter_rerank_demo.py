"""Step 5: metadata filters + reranking.

Run:  .venv\Scripts\python -W ignore step05_filter_rerank_demo.py
"""
import sys

from govtgpt.search import HybridSearcher, rerank

sys.stdout.reconfigure(encoding="utf-8")
s = HybridSearcher()


def show(label, docs, n=5):
    print(f"  {label}")
    for d in docs[:n]:
        m = d.metadata
        print(f"     {m['short']} {m['unit']} {m['section']} (chunk {m['chunk_index']}) "
              f"p{m['page']} [{m['status']}]")


# 1) FILTER: only the DPDP Act -> unrelated IT Act sections cannot appear
q = "What is the penalty if a company fails to protect personal data?"
print(f"\n=== FILTER demo: {q}")
show("no filter      ", s.hybrid(q))
show("DPDP Act only  ", s.hybrid(q, filters={"short": ["DPDP Act 2023"]}))
show("Acts, not struck_down", s.hybrid(q, filters={"doc_type": ["act"],
                                                  "exclude_status": ["struck_down"]}))

# 2) RERANK: take a wide hybrid pool (10), let the LLM re-order, keep top 4
for q in [
    "Within how many hours must an intermediary remove non-consensual intimate images?",
    "What are the duties of a Data Fiduciary?",
]:
    print(f"\n=== RERANK demo: {q}")
    pool = s.hybrid(q, k=10)
    show("hybrid top-4   ", pool, 4)
    show("after rerank   ", rerank(q, pool, top_n=4), 4)
