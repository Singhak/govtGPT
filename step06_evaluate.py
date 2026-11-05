"""Step 6: measure retrieval quality on a gold set (artifacts/eval_set.json).

Run:  .venv\Scripts\python -W ignore step06_evaluate.py            # dense, BM25, hybrid
      .venv\Scripts\python -W ignore step06_evaluate.py --rerank   # also hybrid+rerank (LLM calls)

Metrics (a hit = the gold Section/Rule appears among the retrieved chunks):
  Hit@k  share of questions whose gold section is in the top-k  (recall-like)
  MRR    mean of 1/rank of the first correct result (rewards ranking it HIGH)
"""
import json
import sys

from govtgpt.search import HybridSearcher, rerank

sys.stdout.reconfigure(encoding="utf-8")
ev = json.load(open("artifacts/eval_set.json", encoding="utf-8"))
s = HybridSearcher()
KS = (1, 3, 5, 10)


def sections(docs):
    """Chunk list -> ordered unique (short, section) list."""
    out = []
    for d in docs:
        key = (d.metadata["short"], d.metadata["section"])
        if key not in out:
            out.append(key)
    return out


def rank_of(docs, gold):
    secs = sections(docs)
    return secs.index(gold) + 1 if gold in secs else None


def report(name, ranks):
    n = len(ranks)
    hit = {k: sum(1 for r in ranks if r and r <= k) / n for k in KS}
    mrr = sum(1 / r for r in ranks if r) / n
    print(f"{name:16}" + "".join(f" Hit@{k}={hit[k]:.2f}" for k in KS) + f"  MRR={mrr:.3f}")


methods = {
    "dense": lambda q: s.dense(q, 20),
    "bm25": lambda q: s.sparse(q, 20),
    "hybrid": lambda q: s.hybrid(q, k=20, pool=30),
}
results = {name: [] for name in methods}
for e in ev:
    gold = (e["short"], e["section"])
    for name, fn in methods.items():
        results[name].append(rank_of(fn(e["q"]), gold))

print(f"{len(ev)} questions\n")
for name in methods:
    report(name, results[name])

# Hybrid misses: show them, so we can learn WHY
print("\nHybrid failures (gold section not in top-5):")
for e, r in zip(ev, results["hybrid"]):
    if not r or r > 5:
        print(f"  rank={r} | {e['short']} {e['section']} | {e['q'][:70]}")

if "--rerank" in sys.argv:
    rr = []
    for e in ev:
        gold = (e["short"], e["section"])
        pool = s.hybrid(e["q"], k=10)
        rr.append(rank_of(rerank(e["q"], pool, top_n=10), gold))
    print()
    report("hybrid (pool 10)", [rank_of(s.hybrid(e["q"], k=10), (e["short"], e["section"])) for e in ev])
    report("+ LLM rerank", rr)
