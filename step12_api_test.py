"""Step 12: smoke-test the API in-process (no server needed). Costs ~4 LLM calls.
Run:  .venv\Scripts\python -W ignore step12_api_test.py
"""
import json
import sys
from pathlib import Path

from fastapi.testclient import TestClient

from govtgpt.api import LOG_PATH, app

sys.stdout.reconfigure(encoding="utf-8")
LOG_PATH.unlink(missing_ok=True)
with TestClient(app) as c:                        # context manager runs lifespan (builds graph)
    print("health:", c.get("/health").json())
    for label, payload in [
        ("in-scope", {"question": "What is the penalty for failing to protect personal data under the DPDP Act?", "thread_id": "t1"}),
        ("out-of-scope", {"question": "What is the fine under California's CCPA?", "thread_id": "t2"}),
        ("empty", {"question": "", "thread_id": "t3"}),
        ("too long", {"question": "x" * 600, "thread_id": "t4"}),
        ("bad thread_id", {"question": "What is section 66E?", "thread_id": "../etc"}),
    ]:
        r = c.post("/ask", json=payload)
        b = r.json()
        if r.status_code == 200:
            print(f"{label}: 200 answerable={b['answerable']} refused_by={b['refused_by']!r} "
                  f"cites={[(x['act'], x['section']) for x in b['citations']]} {b['latency_ms']}ms")
        else:
            print(f"{label}: {r.status_code} {str(b)[:110]}")
print("\nlog file lines:")
for line in Path(LOG_PATH).read_text(encoding="utf-8").splitlines():
    d = json.loads(line)
    print({k: d.get(k) for k in ("request_id", "answerable", "refused_by", "queries", "cited", "latency_ms", "error")})
