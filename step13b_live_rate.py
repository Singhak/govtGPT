"""Step 13b: live refusal rate of the numbers check, old vs new rule, on IDENTICAL fresh drafts.
~32 LLM calls. Run:  .venv\Scripts\python -W ignore step13b_live_rate.py
"""
import json, sys, time
from pathlib import Path
from dotenv import load_dotenv
from govtgpt.answer import generate, verify, split_refs
from govtgpt.search import HybridSearcher

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")
s = HybridSearcher()
cache = json.loads(Path("artifacts/drafts_8b.json").read_text(encoding="utf-8"))
MAXFINE = "What is the maximum fine the Data Protection Board can impose?"
plan = [(MAXFINE, 10)] + [(q, 2) for q in cache if q != MAXFINE]


def gen(q, docs):
    for _ in range(4):
        try:
            return generate(q, docs)
        except Exception as e:
            if "PerDay" in str(e):
                sys.exit("DAILY QUOTA exhausted: " + str(e)[:150])
            time.sleep(20)
    raise RuntimeError("gave up")


rows = []
for q, n in plan:
    docs = s.retrieve(q, k=6)
    for i in range(n):
        d = gen(q, docs)
        old = verify(q, docs, d, use_judge=False, ref_aware=False)
        new = verify(q, docs, d, use_judge=False, ref_aware=True)
        rows.append(dict(q=q, answerable=d.answerable, old=old.refused_by, new=new.refused_by,
                         old_w=old.warnings, new_w=new.warnings, answer=d.answer,
                         quotes=[c.quote for c in d.citations], refs=sorted(split_refs(d.answer)[1])))
Path("artifacts/live_rate_13b.json").write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


def summarize(name, rs):
    a = [r for r in rs if r["answerable"]]
    print(f"{name}: n={len(rs)} model-answerable={len(a)} | numbers-refused OLD={sum(r['old']=='numbers' for r in a)}"
          f" NEW={sum(r['new']=='numbers' for r in a)} | other refusals new={sum(r['new'] not in ('','numbers') for r in a)}")


summarize("MAX FINE (10 gens)", [r for r in rows if r["q"] == MAXFINE])
summarize("OTHER 11 Qs (2 gens each)", [r for r in rows if r["q"] != MAXFINE])
print("\nCases still refused by `numbers` under the NEW rule:")
for r in rows:
    if r["answerable"] and r["new"] == "numbers":
        print("-", r["q"][:70], "\n   ", r["new_w"], "\n    refs:", r["refs"], "\n    ANSWER:", r["answer"][:260])
