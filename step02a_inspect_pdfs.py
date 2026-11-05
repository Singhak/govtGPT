"""Step 2a: PDF se raw text nikaalo aur dekho kitna gandha hai.

Chunking se PEHLE raw text dekhna zaroori hai - chunker usi ki
structure par depend karta hai.

Run:  .venv\\Scripts\\python -W ignore step02a_inspect_pdfs.py
"""
import re
import sys
from pathlib import Path

from pypdf import PdfReader

sys.stdout.reconfigure(encoding="utf-8")  # Windows console me Hindi/unicode ke liye

DATA = Path("data")
OUT = Path("data_txt")
OUT.mkdir(exist_ok=True)

for pdf in sorted(DATA.glob("*.pdf")):
    reader = PdfReader(pdf)
    pages = [(p.extract_text() or "") for p in reader.pages]
    text = "\n".join(pages)
    (OUT / f"{pdf.stem}.txt").write_text(text, encoding="utf-8")

    devanagari = len(re.findall(r"[ऀ-ॿ]", text))
    print("=" * 70)
    print(f"{pdf.name}: pages={len(pages)} chars={len(text)} hindi_chars={devanagari}")
    # Section-heading jaisi lines: "43A. Compensation ..." / "4. Due diligence ..."
    heads = re.findall(r"^\s*(\d{1,3}[A-Z]?)\.\s+[A-Z][^\n]{3,60}", text, re.M)
    print("numbered headings found:", len(heads), heads[:12])
    print("--- sample (page 3, first 500 chars) ---")
    print(pages[min(2, len(pages) - 1)][:500])
