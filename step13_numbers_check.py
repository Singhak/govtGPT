"""Step 13: make the numbers check reference-aware, WITHOUT letting wrong numbers through.
OFFLINE (cached 8b drafts, no LLM calls). Compares L2 old (ref_aware=False) vs new (ref_aware=True).
Run:  .venv\Scripts\python -W ignore step13_numbers_check.py
"""
import json
import re
import sys
from pathlib import Path

from dotenv import load_dotenv

from govtgpt.answer import Citation, Draft, numbers_in, split_refs, verify
from govtgpt.search import HybridSearcher

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")
s = HybridSearcher()
cache = json.loads(Path("artifacts/drafts_8b.json").read_text(encoding="utf-8"))
honest = []
for q, raw in cache.items():
    dr = Draft(**raw)
    if dr.answerable and dr.citations:
        honest.append((q, s.retrieve(q, k=6), dr))


def bump(m):
    n = int(m.group(0).replace(",", ""))
    return str(n + 7 if n < 100 else n * 3)


UNIT = re.compile(r"(?:\d[\d,]*|[a-z]+(?:[\s-]+[a-z]+){0,3})\s+(?:days|hours|years|months|crore|lakh)", re.I)
cases = []                                   # (kind, q, docs, draft, should_be_refused)
for i, (q, docs, dr) in enumerate(honest):
    cases.append(("HONEST", q, docs, dr, False))
    new = re.sub(r"\d[\d,]*", bump, dr.answer)
    if new == dr.answer and numbers_in(dr.answer):
        new = dr.answer + " The prescribed limit is 90 days."
    if new != dr.answer:
        cases.append(("T1 wrong number", q, docs, dr.model_copy(update={"answer": new}), True))
    other = honest[(i + 1) % len(honest)][2]
    cases.append(("T2 swapped answer", q, docs, dr.model_copy(update={"answer": other.answer}), True))
    honest_nums = numbers_in(dr.answer)
    for c in dr.citations:
        text = docs[c.passage - 1].page_content
        hit = None
        for m in UNIT.finditer(text):
            wrong = numbers_in(m.group(0)) - honest_nums - numbers_in(q)
            window = text[max(0, m.start() - 250): m.end() + 250]
            if wrong and honest_nums & numbers_in(window):
                hit = (m, window)
                break
        if hit:
            bad = dr.model_copy(update={"answer": "The figure asked about is " + hit[0].group(0).strip() + ".",
                                        "citations": [c.model_copy(update={"quote": hit[1]})]})
            cases.append(("T4 wrong figure in quote", q, docs, bad, True))
            break
    # T5 (new): a provision reference that is NOT in the cited passages (hallucinated section)
    cases.append(("T5 invented section ref", q, docs,
                  dr.model_copy(update={"answer": dr.answer + " This is set out in section 999."}), True))
    # T6 (new): a real section number from a DIFFERENT passage that was not cited
    cited = {c.passage for c in dr.citations}
    cited_digits = {int(x) for p in cited for x in re.findall(r"\d+", docs[p - 1].page_content)}
    for j, d in enumerate(docs, 1):
        if j in cited:
            continue
        sec = re.match(r"\d+", str(d.metadata["section"]))
        if sec and int(sec.group(0)) not in cited_digits and int(sec.group(0)) > 1:
            cases.append(("T6 ref to uncited passage", q, docs,
                          dr.model_copy(update={"answer": dr.answer + f" See also section {sec.group(0)}."}), True))
            break

# R1: replica of the real false refusal ("maximum fine" listing the whole penalty table + section refs)
q = "What is the maximum fine the Data Protection Board can impose?"
docs = s.retrieve(q, k=6)
sched = next(i for i, d in enumerate(docs, 1) if d.metadata["section"] == "44")
passage = docs[sched - 1].page_content
quotes = ["May extend to two hundred and fifty crore rupees.", "May extend to two hundred crore",
          "May extend to fifty crore rupees.", "May extend to ten thousand rupees."]
ok_quotes = [x for x in quotes if re.sub(r"[\W_]+", "", x).lower() in re.sub(r"[\W_]+", "", passage).lower()]
r1 = Draft(answerable=True, citations=[Citation(passage=sched, quote=x) for x in ok_quotes],
           answer="The Schedule lists up to two hundred and fifty crore rupees for failing to take security "
                  "safeguards (section 8), up to ten thousand rupees for breach of duties (section 15), "
                  "and other fines under section 28.")
print("R1 replica: quotes found in passage:", len(ok_quotes), "| digits of refs in passage:",
      {n: bool(re.search(rf"\b{n}\b", passage)) for n in (8, 15, 28)})
cases.append(("R1 real-style refs (honest)", q, docs, r1, False))

kinds = sorted({c[0] for c in cases})
print(f"\n{'case':30}{'n':>4}{'OLD L2 refused':>16}{'NEW L2 refused':>16}   (want: HONEST/R1 = 0, tampered = all)")
for k in kinds:
    sub = [c for c in cases if c[0] == k]
    row = []
    for ra in (False, True):
        row.append(sum(not verify(q_, d_, dr_, check_numbers=True, use_judge=False, ref_aware=ra).answerable
                       for _, q_, d_, dr_, _ in sub))
    print(f"{k:30}{len(sub):>4}{f'{row[0]}/{len(sub)}':>16}{f'{row[1]}/{len(sub)}':>16}")
print("\nref parsing examples:", split_refs("fine under section 15 and 28 is ten thousand rupees"))
for k, q_, d_, dr_, _ in cases:
    if k.startswith(("HONEST", "R1")):
        for ra in (False, True):
            r = verify(q_, d_, dr_, check_numbers=True, use_judge=False, ref_aware=ra)
            if not r.answerable and ra:
                print("STILL REFUSED (new):", k, q_[:60], r.warnings)
