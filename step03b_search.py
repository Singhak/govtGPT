"""Step 3b: first semantic search. Pass your own question as an argument.

Run:  .venv\Scripts\python -W ignore step03b_search.py "your question"
"""
import sys

from govtgpt.store import load_store

sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS = sys.argv[1:] or [
    "What is the penalty if a company fails to protect personal data?",
    "Can I be arrested for posting an offensive message online?",
    "Within how many hours must an intermediary remove non-consensual intimate images?",
    "What is the Section 43A?",
]
store = load_store()
for q in QUESTIONS:
    print(f"\n=== Q: {q}")
    for doc, score in store.similarity_search_with_score(q, k=4):
        m = doc.metadata
        print(f"  {score:.3f} | {m['short']} {m['unit']} {m['section']} "
              f"p{m['page']} [{m['status']}] {m['title'][:50]}")
