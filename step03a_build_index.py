"""Step 3a: chunks -> embeddings -> Qdrant.  Run once.

Run:  .venv\Scripts\python -W ignore step03a_build_index.py
"""
import sys

from govtgpt.ingest import ingest
from govtgpt.store import COLLECTION, DIM, EMBED_MODEL, build_index, get_client

sys.stdout.reconfigure(encoding="utf-8")

docs = ingest()
print(f"{len(docs)} chunks | model={EMBED_MODEL} | dim={DIM}")
build_index(docs)
client = get_client()
print(f"\nStored points in '{COLLECTION}':", client.count(COLLECTION).count)
