"""Step 11: does routing `numbers` refusals to a regenerate-with-feedback node help?
Compares, over REPEATS runs of the 16 hard questions: one-shot | graph (fix off) | graph (fix on).
Run:  .venv\Scripts\python -W ignore step11_fix_numbers.py
"""
import json, sys, time
from dotenv import load_dotenv
from govtgpt.answer import answer
from govtgpt.graph import ask, build_graph
from govtgpt.search import HybridSearcher

load_dotenv()
sys.stdout.reconfigure(encoding="utf-8")
REPEATS = 2
s = HybridSearcher()
g_off, g_on = build_graph(s, fix_numbers=False), build_graph(s, fix_numbers=True)
hard = json.load(open("artifacts/eval_hard.json", encoding="utf-8"))["in_scope"]


def retry(fn, *a):
    for _ in range(5):
        try:
            return fn(*a)
        except Exception as e:
            if "PerDay" in str(e):
                sys.exit("DAILY QUOTA exhausted, stopping: " + str(e)[:200])
            print("   (retry:", str(e)[:100], ")")
            time.sleep(25)
    raise RuntimeError("gave up")


def gold_ok(r, gold):
    cited = {(r.sources[c.passage - 1].metadata["short"], r.sources[c.passage - 1].metadata["section"])
             for c in r.citations}
    return bool(cited & {tuple(x) for x in gold})


tot = {k: [0, 0] for k in ("one", "off", "on")}
fixed = fix_fired = 0
for rep in range(REPEATS):
    for i, it in enumerate(hard):
        q = it["q"]
        r1 = retry(lambda: answer(q, s.retrieve(q, k=6)))
        r2 = retry(ask, g_off, q, f"off-{rep}-{i}")["result"]
        st = retry(ask, g_on, q, f"on-{rep}-{i}")
        r3 = st["result"]
        for k, r in (("one", r1), ("off", r2), ("on", r3)):
            tot[k][0] += r.answerable
            tot[k][1] += r.answerable and gold_ok(r, it["gold"])
        if any(l.startswith("fix:") for l in st["log"]):
            fix_fired += 1
            fixed += r3.answerable
            print(f"[rep{rep} q{i}] fix fired -> answerable={r3.answerable} refused_by={r3.refused_by!r} | {q[:60]}")
n = REPEATS * len(hard)
print(f"\nSUMMARY over {n} question-runs (answered / answered+cites gold):")
for k, name in (("one", "one-shot"), ("off", "graph fix OFF"), ("on", "graph fix ON")):
    print(f"  {name:14s} {tot[k][0]}/{n}  {tot[k][1]}/{n}")
print(f"fix node fired {fix_fired} times; ended answerable {fixed} times")
