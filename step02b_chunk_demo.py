"""Step 2b: ingestion chalao aur chunks ko khud verify karo.

Run:  .venv\\Scripts\\python -W ignore step02b_chunk_demo.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

from govtgpt.ingest import CONFIGS, ingest, load_pages, split_into_units

sys.stdout.reconfigure(encoding="utf-8")

# --- 1) Har document me kitne units (sections/rules) mile? ------------------
print("UNITS PER DOCUMENT")
for cfg in CONFIGS:
    units = split_into_units(load_pages(Path("data") / cfg.file), cfg)
    nums = [u.number for u in units if not u.is_schedule]
    sched = sum(u.is_schedule for u in units)
    print(f"  {cfg.short:16} {len(nums):3} {cfg.unit.lower()}s + {sched} schedules"
          f" | first={nums[:3]} last={nums[-3:]}")

# --- 2) Poora ingest + chunk stats ------------------------------------------
docs = ingest()
sizes = [len(d.page_content) for d in docs]
print(f"\nTOTAL CHUNKS: {len(docs)} | chars min/avg/max = "
      f"{min(sizes)}/{sum(sizes)//len(sizes)}/{max(sizes)}")
print("per doc:", dict(Counter(d.metadata["short"] for d in docs)))

# --- 3) Spot-check: jaane-pehchaane sections ----------------------------------
def show(short, section, n=450):
    for d in docs:
        m = d.metadata
        if m["short"] == short and m["section"] == section and m["chunk_index"] == 0:
            print(f"\n--- {short} {section} | page {m['page']} | status={m['status']} "
                  f"| chunks={m['chunk_total']} ---")
            print(d.page_content[:n])
            return
    print(f"\n!!! NOT FOUND: {short} {section}")

show("IT Act 2000", "43A")
show("IT Act 2000", "66A")
show("DPDP Act 2023", "8")
show("IT Rules 2021", "4")
show("DPDP Rules 2025", "3")

# --- 4) Save karo (agle step me embeddings isi se banenge) -------------------
out = Path("artifacts"); out.mkdir(exist_ok=True)
with open(out / "chunks.jsonl", "w", encoding="utf-8") as f:
    for d in docs:
        f.write(json.dumps({"text": d.page_content, "metadata": d.metadata},
                           ensure_ascii=False) + "\n")
print(f"\nSaved {len(docs)} chunks -> artifacts/chunks.jsonl")
