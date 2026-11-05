"""Step 13c: re-verify the SAVED live drafts (artifacts/live_rate_13b.json) with the refined rule. No LLM calls."""
import json, sys
from collections import Counter
from dotenv import load_dotenv
from govtgpt.answer import Citation, Draft, _norm, verify
from govtgpt.search import HybridSearcher

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")
s = HybridSearcher()
rows = json.load(open("artifacts/live_rate_13b.json", encoding="utf-8"))
docs_for = {}
stats = {"fine": Counter(), "other": Counter()}
for r in rows:
    q = r["q"]
    docs = docs_for.setdefault(q, s.retrieve(q, k=6))
    cites = []
    for qt in r["quotes"]:
        p = next((i for i, d in enumerate(docs, 1) if all(_norm(f) in _norm(d.page_content) for f in [qt] if _norm(f))), None)
        if p:
            cites.append(Citation(passage=p, quote=qt))
    d = Draft(answerable=r["answerable"], answer=r["answer"], citations=cites)
    key = "fine" if q.startswith("What is the maximum fine") else "other"
    if not r["answerable"]:
        continue
    stats[key]["n"] += 1
    stats[key]["old numbers-refused"] += verify(q, docs, d, use_judge=False, ref_aware=False).refused_by == "numbers"
    new = verify(q, docs, d, use_judge=False, ref_aware=True)
    stats[key]["new numbers-refused"] += new.refused_by == "numbers"
    stats[key]["new other refusal"] += new.refused_by not in ("", "numbers")
    if new.refused_by:
        print("STILL REFUSED:", q[:60], new.refused_by, new.warnings)
for k, c in stats.items():
    print(k, dict(c))
