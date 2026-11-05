"""Step 8b: can we catch "real quote + wrong answer"?

1. Generate honest drafts once for in-scope questions (cached).
2. Tamper them: T1 change the numbers, T2 swap in another question's answer, keep the real quote.
3. Measure with: L1 only (Step 8), +L2 numbers (code), +L3 LLM judge.
   Also measure false alarms on the honest drafts.

Run:  .venv\\Scripts\\python -W ignore step08b_faithfulness.py
"""
import json
import re
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

from govtgpt.answer import Draft, generate, numbers_in, verify
from govtgpt.search import HybridSearcher

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")

QUESTIONS = [
    "What is the penalty for failing to protect personal data under the DPDP Act?",
    "Within how many hours must an intermediary remove non-consensual intimate images?",
    "Can a Data Fiduciary process children's data without parental consent?",
    "What is the maximum penalty for failing to notify the Board of a personal data breach?",
    "Within how many days must the Grievance Officer dispose of a complaint?",
    "What is the punishment for identity theft under the IT Act?",
    "What is the punishment for cheating by personation using a computer resource?",
    "What is the punishment for cyber terrorism?",
    "What is the punishment for publishing obscene material in electronic form?",
    "What is the penalty for breach of obligations relating to children under the DPDP Act?",
    "What is the punishment for violation of privacy by capturing a private image?",
    "What is the penalty for failure to furnish information to the Board?",
]

CACHE = Path("artifacts/drafts_8b.json")
s = HybridSearcher()
cache = json.loads(CACHE.read_text(encoding="utf-8")) if CACHE.exists() else {}

# ---- 1) honest drafts ---------------------------------------------------------
items = []                         # (question, docs, draft)
for q in QUESTIONS:
    docs = s.retrieve(q, k=6)
    if q not in cache:
        cache[q] = generate(q, docs).model_dump()
        CACHE.write_text(json.dumps(cache, ensure_ascii=False, indent=1), encoding="utf-8")
    items.append((q, docs, Draft(**cache[q])))
honest = [(q, d, dr) for q, d, dr in items if dr.answerable and dr.citations]
print(f"drafts: {len(items)} | answerable: {len(honest)}")
for q, _, dr in items:
    print(f"  {'OK ' if dr.answerable else 'REF'} {q[:70]} | nums={sorted(numbers_in(dr.answer))}")


# ---- 2) tampered drafts --------------------------------------------------------
def bump(m):
    n = int(m.group(0).replace(",", ""))
    return str(n + 7 if n < 100 else n * 3)


UNIT_PHRASE = re.compile(
    r"(?:\d[\d,]*|[a-z]+(?:[\s-]+[a-z]+){0,3})\s+(?:days|hours|years|months|crore|lakh)", re.I)
tampered = []                      # (kind, question, docs, bad_draft)
for i, (q, docs, dr) in enumerate(honest):
    # T1: change every digit-number; for word-numbers prepend a clearly different figure
    new = re.sub(r"\d[\d,]*", bump, dr.answer)
    if new == dr.answer and numbers_in(dr.answer):
        new = dr.answer + " The prescribed limit is 90 days."
    if new != dr.answer:
        tampered.append(("T1 wrong number", q, docs, dr.model_copy(update={"answer": new})))
    # T2: answer taken from a DIFFERENT question, citations (real quotes) kept
    other = honest[(i + 1) % len(honest)][2]
    tampered.append(("T2 swapped answer", q, docs, dr.model_copy(update={"answer": other.answer})))
    # T3: meaning flipped without any number change
    flip = dr.answer
    for a, b in ((" must ", " need not "), (" shall ", " shall not "), (" may ", " may not "),
                 (" can ", " cannot "), ("Yes", "No")):
        if a in flip:
            flip = flip.replace(a, b, 1)
            break
    if flip != dr.answer:
        tampered.append(("T3 flipped meaning", q, docs, dr.model_copy(update={"answer": flip})))
    # T4: the WRONG figure is present in the cited quote (the quote window holds both the right
    #     and the wrong figure), so the numbers check has nothing to complain about.
    honest_nums = numbers_in(dr.answer)
    done = False
    for c in dr.citations:
        text = docs[c.passage - 1].page_content
        for m in UNIT_PHRASE.finditer(text):
            wrong = numbers_in(m.group(0)) - honest_nums - numbers_in(q)
            window = text[max(0, m.start() - 250): m.end() + 250]
            if wrong and honest_nums & numbers_in(window):
                bad = dr.model_copy(update={
                    "answer": "The figure asked about is " + m.group(0).strip() + ".",
                    "citations": [c.model_copy(update={"quote": window})]})
                tampered.append(("T4 wrong figure in quote", q, docs, bad))
                done = True
                break
        if done:
            break

print(f"\ntampered cases: {len(tampered)}  "
      f"({', '.join(f'{k}={sum(t[0] == k for t in tampered)}' for k in sorted({t[0] for t in tampered}))})")


def refused(q, docs, dr, **kw):
    for attempt in range(4):
        try:
            return not verify(q, docs, dr, **kw).answerable
        except Exception as e:      # rate limit
            if attempt == 3:
                raise
            time.sleep(20 * (attempt + 1))


CONFIGS = {
    "L1 quote only (Step 8)": dict(check_numbers=False, use_judge=False),
    "L1+L2 numbers (code)": dict(check_numbers=True, use_judge=False),
    "L1+L2+L3 LLM judge": dict(check_numbers=True, use_judge=True),
}

# ---- 3) results ----------------------------------------------------------------
print("\nCAUGHT (refused) among TAMPERED answers, and FALSE ALARMS among HONEST answers")
kinds = sorted({t[0] for t in tampered})
print(f"{'config':26}" + "".join(f"{k:>22}" for k in kinds) + f"{'false alarms':>16}")
false_alarm_cases = {}
for name, kw in CONFIGS.items():
    row = f"{name:26}"
    for k in kinds:
        sub = [t for t in tampered if t[0] == k]
        caught = sum(refused(q, d, dr, **kw) for _, q, d, dr in sub)
        row += f"{f'{caught}/{len(sub)}':>22}"
    fa = [q for q, d, dr in honest if refused(q, d, dr, **kw)]
    false_alarm_cases[name] = fa
    row += f"{f'{len(fa)}/{len(honest)}':>16}"
    print(row)

for name, fa in false_alarm_cases.items():
    for q in fa:
        r = verify(q, *[x for x in honest if x[0] == q][0][1:], **CONFIGS[name])
        print(f"\nfalse alarm [{name}]: {q}\n   {r.warnings}")
