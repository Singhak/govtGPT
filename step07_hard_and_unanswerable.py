"""Step 7: harder questions + questions the documents cannot answer.

Run:  .venv\Scripts\python -W ignore step07_hard_and_unanswerable.py
"""
import json
import re
import statistics
import sys

from govtgpt.search import HybridSearcher

sys.stdout.reconfigure(encoding="utf-8")
data = json.load(open("artifacts/eval_hard.json", encoding="utf-8"))
chunks = [json.loads(l) for l in open("artifacts/chunks.jsonl", encoding="utf-8")]

# 0) verify gold against the real indexed text
bad = 0
for e in data["in_scope"]:
    txt = " ".join(c["text"] for c in chunks for g in e["gold"]
                   if (c["metadata"]["short"], c["metadata"]["section"]) == tuple(g))
    if not txt or not re.search(e["must"], txt, re.I):
        print("GOLD NOT VERIFIED:", e["gold"], "|", e["q"][:60]); bad += 1
print(f"gold check: {len(data['in_scope'])} questions, {bad} unverified\n")

s = HybridSearcher()


def secs(docs):
    out = []
    for d in docs:
        k = (d.metadata["short"], d.metadata["section"])
        if k not in out:
            out.append(k)
    return out


def best_rank(docs, gold):
    sc = secs(docs)
    r = [sc.index(tuple(g)) + 1 for g in gold if tuple(g) in sc]
    return min(r) if r else None


methods = {"dense": lambda q: s.dense(q, 20), "bm25": lambda q: s.sparse(q, 20),
           "hybrid": lambda q: s.hybrid(q, k=20, pool=30)}
ranks = {m: [] for m in methods}
for e in data["in_scope"]:
    for m, fn in methods.items():
        ranks[m].append(best_rank(fn(e["q"]), e["gold"]))

n = len(data["in_scope"])
print(f"HARD in-scope questions ({n})")
for m in methods:
    r = ranks[m]
    print(f"{m:8}" + "".join(f" Hit@{k}={sum(1 for x in r if x and x <= k)/n:.2f}" for k in (1, 3, 5, 10))
          + f"  MRR={sum(1/x for x in r if x)/n:.3f}")

print("\nHybrid misses (gold not in top-5):")
for e, r in zip(data["in_scope"], ranks["hybrid"]):
    if not r or r > 5:
        print(f"  rank={r} | {e['q'][:62]} | top3={secs(s.hybrid(e['q'], k=10))[:3]}")

# 1) Can a score tell 'answerable' from 'not in the documents'?
def top_cosine(q):
    return s.store.similarity_search_with_score_by_vector(s._embed_query(q), k=1)[0][1]

def top_bm25(q):
    import numpy as np
    from govtgpt.search import _tok
    return float(np.max(s.bm25.get_scores(_tok(q))))

easy = [e["q"] for e in json.load(open("artifacts/eval_set.json", encoding="utf-8"))]
ins = [e["q"] for e in data["in_scope"]] + easy
outs = data["out_of_scope"]
ic = [top_cosine(q) for q in ins]; oc = [top_cosine(q) for q in outs]
ib = [top_bm25(q) for q in ins];   ob = [top_bm25(q) for q in outs]

print(f"\nTOP-1 DENSE COSINE   in-scope ({len(ic)}): min={min(ic):.3f} median={statistics.median(ic):.3f}"
      f" | out-of-scope ({len(oc)}): median={statistics.median(oc):.3f} max={max(oc):.3f}")
print(f"TOP-1 BM25 SCORE     in-scope: min={min(ib):.1f} median={statistics.median(ib):.1f}"
      f" | out-of-scope: median={statistics.median(ob):.1f} max={max(ob):.1f}")
print("\nOut-of-scope questions: what did we retrieve anyway?")
for q, c, b in zip(outs, oc, ob):
    top = secs(s.hybrid(q, k=3))[:2]
    print(f"  cos={c:.3f} bm25={b:5.1f} | {q[:52]:52} | {top}")
print("\nLowest in-scope cosine scores (hardest to separate):")
for sc, q in sorted(zip(ic, ins))[:4]:
    print(f"  cos={sc:.3f} | {q[:70]}")
